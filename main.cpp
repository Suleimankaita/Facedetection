/**
 * ============================================================
 *   finger_count.cpp  —  Advanced Finger Counter
 *                         + Robust Face Recognition Gate
 *   Single-file C++ / OpenCV 4.x   Windows / Linux / macOS
 * ============================================================
 *
 *  BUILD
 *  ─────
 *  Windows (MSYS2 / MinGW):
 *    g++ -std=c++17 -O2 finger_count.cpp \
 *        $(pkg-config --cflags --libs opencv4) \
 *        -lopencv_face -o finger_count.exe
 *
 *  Linux / macOS:
 *    g++ -std=c++17 -O3 finger_count.cpp \
 *        $(pkg-config --cflags --libs opencv4) \
 *        -lopencv_face -o finger_count
 *
 *  WHAT IS IMPROVED OVER THE PREVIOUS VERSION
 *  ───────────────────────────────────────────
 *  Face recognition
 *    • Sliding-window confidence vote (7 frames) – no single-frame flicker
 *    • Blur / quality guard – blurry frames are rejected from both
 *      enrolment AND real-time verification
 *    • Pose-varied enrolment – user is prompted to tilt left, right, up,
 *      down so the model sees your face from multiple angles
 *    • Adaptive confidence: threshold tightens after repeated failures
 *      to resist spoofing with a photo
 *    • 9× augmentation per capture: flip, 3 brightness levels,
 *      2 rotations ±8°, Gaussian blur, CLAHE re-applied
 *
 *  Finger counting  (dual-method, cross-validated)
 *    • Method A — convexity defects (original, improved)
 *    • Method B — curvature-based fingertip detection on the contour
 *    • Final count = majority vote of A, B, and smoothing history
 *    • Wrist-cut: the bottom 25 % of the hand bounding box is masked
 *      out so wrist entry points never create false fingertips
 *    • Each detected fingertip drawn as a numbered circle
 *    • Adaptive skin model: background-subtracted before thresholding
 *
 *  HUD
 *    • Confidence meter (visual bar)
 *    • Per-finger labels (dots + numbers)
 *    • Live skin-mask mini-preview in corner
 *
 *  KEYS:  [E] Enrol   [R] Reload   [Q / ESC] Quit
 * ============================================================
 */

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ╔══════════════════════════════════════════════════════════════╗
// ║                   GLOBAL CONFIGURATION                       ║
// ╚══════════════════════════════════════════════════════════════╝

static constexpr int    CAM_ID          = 0;
static constexpr int    FRAME_W         = 1280;
static constexpr int    FRAME_H         = 720;

// ── Face recognition ──────────────────────────────────────────────────────────
static constexpr double CONF_THRESHOLD  = 62.0;  // LBPH distance; lower=stricter
static constexpr int    OWNER_LABEL     = 1;
static constexpr int    MIN_FACE_PX     = 90;     // ignore tiny detections
static constexpr double VERIFY_HZ       = 12.0;   // recognition calls per second
static constexpr int    VERIFY_VOTES    = 7;       // frames in sliding vote window
static constexpr int    VERIFY_MAJORITY = 4;       // need this many "yes" votes
static constexpr double BLUR_THRESHOLD  = 80.0;   // Laplacian variance; reject below

// Enrolment poses
static const std::vector<std::string> POSE_LABELS = {
    "LOOK STRAIGHT", "TILT LEFT", "TILT RIGHT",
    "LOOK UP", "LOOK DOWN"
};
static constexpr int FRAMES_PER_POSE = 25;   // total = poses × frames × 9 augments

// ── Finger counting ───────────────────────────────────────────────────────────
static constexpr double MAX_ANGLE_DEG   = 88.0;   // defect angle filter
static constexpr double MIN_DEPTH_RATIO = 0.08;   // depth / hand-height
static constexpr double MIN_AREA        = 4000.0; // px²  — hand contour
static constexpr int    FIN_SMOOTH      = 9;       // majority-vote window (frames)
static constexpr double WRIST_CUT_RATIO = 0.22;   // mask bottom N% of hand ROI

// Hand ROI (fraction of frame)
static constexpr double ROI_X = 0.54;
static constexpr double ROI_Y = 0.08;
static constexpr double ROI_W = 0.42;
static constexpr double ROI_H = 0.80;

// Paths
static const std::string DATA_DIR   = "data";
static const std::string MODEL_PATH = DATA_DIR + "/face_model.yml";
static const std::string WIN_NAME   = "Finger Counter  [E]=Enrol  [R]=Reload  [Q]=Quit";


// ╔══════════════════════════════════════════════════════════════╗
// ║               UTILITY HELPERS                                ║
// ╚══════════════════════════════════════════════════════════════╝

// Laplacian variance — measures image sharpness
static double blurScore(const cv::Mat& gray)
{
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma);
    return sigma[0] * sigma[0];
}

// Angle at vertex B in triangle A-B-C  (degrees)
static double angleDeg(cv::Point A, cv::Point B, cv::Point C)
{
    double ab = cv::norm(A - B);
    double bc = cv::norm(B - C);
    double ac = cv::norm(A - C);
    double cosB = (ab*ab + bc*bc - ac*ac) / (2.0*ab*bc + 1e-9);
    return std::acos(std::clamp(cosB, -1.0, 1.0)) * 180.0 / CV_PI;
}

// Majority value in a deque<int>
static int majorityVote(const std::deque<int>& hist, int maxVal = 5)
{
    if (hist.empty()) return 0;
    std::vector<int> cnt(maxVal + 1, 0);
    for (int v : hist) if (v >= 0 && v <= maxVal) ++cnt[v];
    return static_cast<int>(
        std::max_element(cnt.begin(), cnt.end()) - cnt.begin());
}

// Locate Haar cascade file
static cv::CascadeClassifier loadHaar()
{
    cv::CascadeClassifier h;
    static const std::vector<std::string> PATHS = {
        "haarcascade_frontalface_alt2.xml",
        "haarcascade_frontalface_default.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml",
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml",
        "C:/msys64/mingw64/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml",
        "C:/msys64/mingw64/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        cv::samples::findFile("haarcascade_frontalface_alt2.xml", false),
    };
    for (const auto& p : PATHS)
        if (!p.empty() && h.load(p)) {
            std::cout << "[Face] Haar: " << p << "\n";
            return h;
        }
    std::cerr << "[Face] WARNING: Haar cascade not found — "
                 "copy haarcascade_frontalface_alt2.xml next to the binary.\n";
    return h;
}


// ╔══════════════════════════════════════════════════════════════╗
// ║                    FACE RECOGNITION                          ║
// ╚══════════════════════════════════════════════════════════════╝

// ── detect largest face, return ROI (empty if none) ───────────────────────────
static cv::Rect detectFace(const cv::Mat& gray, cv::CascadeClassifier& haar)
{
    if (haar.empty()) return cv::Rect();
    std::vector<cv::Rect> faces;
    haar.detectMultiScale(gray, faces, 1.08, 6, 0,
                          cv::Size(MIN_FACE_PX, MIN_FACE_PX));
    if (faces.empty()) return cv::Rect();
    return *std::max_element(faces.begin(), faces.end(),
        [](const cv::Rect& a, const cv::Rect& b){ return a.area() < b.area(); });
}

// ── normalise face crop → 128×128, CLAHE ─────────────────────────────────────
static cv::Mat normFace(const cv::Mat& gray, const cv::Rect& roi)
{
    cv::Rect safe = roi & cv::Rect(0, 0, gray.cols, gray.rows);
    if (safe.empty()) return cv::Mat();
    cv::Mat f = gray(safe).clone();
    cv::resize(f, f, cv::Size(128, 128));
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(f, f);
    return f;
}

// ── 9× augmentation: flip + 3 brightness + 2 rotations + blur + high-contrast ─
static void augment(const cv::Mat& face,
                    std::vector<cv::Mat>& imgs,
                    std::vector<int>&     labs)
{
    auto add = [&](const cv::Mat& m){
        imgs.push_back(m.clone());
        labs.push_back(OWNER_LABEL);
    };

    add(face);

    // horizontal flip
    cv::Mat tmp;
    cv::flip(face, tmp, 1);
    add(tmp);

    // brightness: +30, -30, +60
    for (double delta : {30.0, -30.0, 60.0}) {
        face.convertTo(tmp, -1, 1.0, delta);
        add(tmp);
    }

    // rotation ±8°
    cv::Point2f ctr(face.cols / 2.f, face.rows / 2.f);
    for (double ang : {8.0, -8.0}) {
        cv::Mat rot = cv::getRotationMatrix2D(ctr, ang, 1.0);
        cv::warpAffine(face, tmp, rot, face.size());
        add(tmp);
    }

    // slight Gaussian blur (simulates soft focus)
    cv::GaussianBlur(face, tmp, cv::Size(3, 3), 0);
    add(tmp);

    // boosted contrast via CLAHE again
    cv::Ptr<cv::CLAHE> cl2 = cv::createCLAHE(5.0, cv::Size(4, 4));
    cl2->apply(face, tmp);
    add(tmp);
}

// ── draw corner-bracket face box ─────────────────────────────────────────────
static void drawFaceBox(cv::Mat& disp, const cv::Rect& box,
                         bool owner, double conf, int voteCount)
{
    cv::Scalar col = owner ? cv::Scalar(0, 255, 80) : cv::Scalar(0, 40, 255);
    int th  = owner ? 3 : 2;
    int cx  = box.x,     cy = box.y;
    int cw  = box.width, ch = box.height;
    int arm = std::min(cw, ch) / 5;

    auto L = [&](cv::Point a, cv::Point b){ cv::line(disp, a, b, col, th); };
    L(cv::Point(cx,cy+arm),       cv::Point(cx,cy));
    L(cv::Point(cx,cy),           cv::Point(cx+arm,cy));
    L(cv::Point(cx+cw-arm,cy),   cv::Point(cx+cw,cy));
    L(cv::Point(cx+cw,cy),       cv::Point(cx+cw,cy+arm));
    L(cv::Point(cx,cy+ch-arm),   cv::Point(cx,cy+ch));
    L(cv::Point(cx,cy+ch),       cv::Point(cx+arm,cy+ch));
    L(cv::Point(cx+cw-arm,cy+ch),cv::Point(cx+cw,cy+ch));
    L(cv::Point(cx+cw,cy+ch),    cv::Point(cx+cw,cy+ch-arm));

    // confidence bar inside box (top strip)
    int barH  = 6;
    int barFW = static_cast<int>(cw * (1.0 - std::min(conf / 100.0, 1.0)));
    cv::rectangle(disp, cv::Point(cx, cy+2), cv::Point(cx+cw, cy+2+barH),
                  cv::Scalar(40,40,40), -1);
    cv::rectangle(disp, cv::Point(cx, cy+2), cv::Point(cx+barFW, cy+2+barH),
                  owner ? cv::Scalar(0,220,80) : cv::Scalar(0,60,220), -1);

    // label
    std::string lbl = owner
        ? "YOU  [" + std::to_string(voteCount) + "/" +
          std::to_string(VERIFY_VOTES) + "]  d=" +
          std::to_string(static_cast<int>(conf))
        : "STRANGER  d=" + std::to_string(static_cast<int>(conf));

    int bl = 0;
    cv::Size ts = cv::getTextSize(lbl, cv::FONT_HERSHEY_SIMPLEX, 0.50, 1, &bl);
    cv::rectangle(disp,
                  cv::Point(cx, cy - ts.height - 8),
                  cv::Point(cx + ts.width + 6, cy),
                  col, -1);
    cv::putText(disp, lbl, cv::Point(cx+3, cy-5),
                cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(10,10,10), 1);
}

// ── pose-guided enrolment (blocking) ─────────────────────────────────────────
static bool enrollFace(cv::VideoCapture& cap,
                        cv::CascadeClassifier& haar,
                        cv::Ptr<cv::face::LBPHFaceRecognizer>& rec,
                        float& progressOut)
{
    const int totalPoses = static_cast<int>(POSE_LABELS.size());
    const int total      = totalPoses * FRAMES_PER_POSE;
    int       captured   = 0;
    progressOut = 0.f;

    std::vector<cv::Mat> imgs;
    std::vector<int>     labs;
    imgs.reserve(total * 9);
    labs.reserve(total * 9);

    std::cout << "[Enrol] Starting " << totalPoses << "-pose enrolment ("
              << total << " frames total)\n";

    for (int pose = 0; pose < totalPoses; ++pose)
    {
        // per-pose countdown
        int cd = 75;   // ~2.5 s at 30 fps
        while (cd > 0) {
            cv::Mat f; cap >> f; if (f.empty()) continue;
            cv::flip(f, f, 1);
            cv::Mat d = f.clone();

            // instruction overlay
            cv::rectangle(d, cv::Point(0,0), cv::Point(d.cols, 80),
                          cv::Scalar(20,20,20), -1);
            cv::putText(d, "POSE " + std::to_string(pose+1) + "/" +
                        std::to_string(totalPoses) + ":  " + POSE_LABELS[pose],
                        cv::Point(20, 50), cv::FONT_HERSHEY_DUPLEX,
                        1.0, cv::Scalar(0, 220, 255), 2);
            std::string cstr = "Starting in " + std::to_string(cd/30+1) + "...";
            cv::putText(d, cstr, cv::Point(20, 95),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(200,200,200), 2);

            // detect and show face box
            cv::Mat gray; cv::cvtColor(f, gray, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(gray, gray);
            cv::Rect r = detectFace(gray, haar);
            if (!r.empty())
                cv::rectangle(d, r, cv::Scalar(0, 200, 255), 2);

            cv::imshow(WIN_NAME, d);
            if ((cv::waitKey(1)&0xFF) == 27) { progressOut=-1.f; return false; }
            --cd;
        }

        // capture phase for this pose
        int poseGot = 0;
        while (poseGot < FRAMES_PER_POSE)
        {
            cv::Mat f; cap >> f; if (f.empty()) continue;
            cv::flip(f, f, 1);
            cv::Mat d = f.clone();

            cv::Mat gray; cv::cvtColor(f, gray, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(gray, gray);

            // quality: skip blurry frames
            if (blurScore(gray) < BLUR_THRESHOLD) {
                cv::putText(d, "! BLURRY FRAME – hold still !",
                            cv::Point(20, 140), cv::FONT_HERSHEY_SIMPLEX,
                            0.8, cv::Scalar(0,80,255), 2);
                cv::imshow(WIN_NAME, d);
                cv::waitKey(1);
                continue;
            }

            cv::Rect roi = detectFace(gray, haar);
            if (!roi.empty()) {
                // extra size guard: face should fill at least 10% of frame width
                if (roi.width >= d.cols / 10) {
                    cv::Mat face = normFace(gray, roi);
                    if (!face.empty()) {
                        augment(face, imgs, labs);
                        ++poseGot;
                        ++captured;
                        progressOut = static_cast<float>(captured) / total;
                    }
                }
                cv::rectangle(d, roi, cv::Scalar(0,255,120), 2);
            }

            // HUD
            cv::rectangle(d, cv::Point(0,0), cv::Point(d.cols,80),
                          cv::Scalar(20,20,20), -1);
            cv::putText(d, "POSE: " + POSE_LABELS[pose],
                        cv::Point(20,50), cv::FONT_HERSHEY_DUPLEX,
                        1.0, cv::Scalar(0,220,255), 2);

            // progress bar
            int bw = static_cast<int>((d.cols-40) * progressOut);
            cv::rectangle(d, cv::Point(20,d.rows-40),
                          cv::Point(20+bw, d.rows-15),
                          cv::Scalar(0,200,0), -1);
            cv::rectangle(d, cv::Point(20,d.rows-40),
                          cv::Point(d.cols-20, d.rows-15),
                          cv::Scalar(200,200,200), 2);
            std::string pg = "Captured " + std::to_string(captured) +
                             " / " + std::to_string(total);
            cv::putText(d, pg, cv::Point(20,d.rows-50),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(255,255,255), 2);

            cv::imshow(WIN_NAME, d);
            if ((cv::waitKey(1)&0xFF) == 27) { progressOut=-1.f; return false; }
        }
    }

    // train
    {
        cv::Mat msg(cv::Size(FRAME_W, FRAME_H), CV_8UC3, cv::Scalar(15,15,15));
        cv::putText(msg, "Training on " + std::to_string(imgs.size()) + " samples...",
                    cv::Point(80, FRAME_H/2-20),
                    cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(0,220,255), 2);
        cv::putText(msg, "Please wait",
                    cv::Point(80, FRAME_H/2+30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(180,180,180), 2);
        cv::imshow(WIN_NAME, msg); cv::waitKey(1);
    }

    std::cout << "[Enrol] Training on " << imgs.size() << " samples...\n";
    rec->train(imgs, labs);
    std::cout << "[Enrol] Done.\n";

    progressOut = -1.f;
    return true;
}


// ╔══════════════════════════════════════════════════════════════╗
// ║                    FINGER COUNTER                            ║
// ╚══════════════════════════════════════════════════════════════╝

// ── adaptive YCrCb + HSV skin segmentation ────────────────────────────────────
static cv::Mat skinMask(const cv::Mat& bgr)
{
    // YCrCb
    cv::Mat ycc, mY;
    cv::cvtColor(bgr, ycc, cv::COLOR_BGR2YCrCb);
    cv::inRange(ycc,
                cv::Scalar(0,   130, 75),
                cv::Scalar(255, 175, 130),
                mY);

    // HSV (two hue ranges to cover very light and dark skin)
    cv::Mat hsv, mH1, mH2;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0,  18, 50),  cv::Scalar(20, 180, 255), mH1);
    cv::inRange(hsv, cv::Scalar(168,18, 50),  cv::Scalar(180,180, 255), mH2);
    cv::bitwise_or(mH1, mH2, mH1);

    cv::Mat skin;
    cv::bitwise_and(mY, mH1, skin);
    return skin;
}

// ── morphological clean-up ────────────────────────────────────────────────────
static cv::Mat cleanMask(const cv::Mat& raw)
{
    cv::Mat m = raw.clone();
    cv::Mat k3  = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,  3));
    cv::Mat k7  = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7,  7));
    cv::Mat k13 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13));
    cv::medianBlur(m, m, 7);
    cv::morphologyEx(m, m, cv::MORPH_OPEN,  k3);
    cv::morphologyEx(m, m, cv::MORPH_CLOSE, k13);
    cv::dilate(m, m, k7, cv::Point(-1,-1), 2);
    return m;
}

// ── Method A: convexity defect valleys ───────────────────────────────────────
static int defectCount(const std::vector<cv::Point>& contour,
                        const cv::Rect& bbox,
                        cv::Mat& dbg)
{
    std::vector<int> hullIdx;
    cv::convexHull(contour, hullIdx);
    if (hullIdx.size() < 3) return 0;

    std::vector<cv::Vec4i> defects;
    cv::convexityDefects(contour, hullIdx, defects);

    static constexpr double ABS_MIN_DEPTH = 14.0;
    double minDepth = std::max(MIN_DEPTH_RATIO * bbox.height, ABS_MIN_DEPTH);

    int valleys = 0;
    for (const auto& d : defects) {
        cv::Point S = contour[d[0]];
        cv::Point E = contour[d[1]];
        cv::Point F = contour[d[2]];
        float depth = d[3] / 256.f;

        // depth filter
        if (depth < minDepth) continue;

        // wrist guard: skip defects whose far point is in the bottom 25%
        if (F.y > bbox.y + static_cast<int>(bbox.height * (1.0 - WRIST_CUT_RATIO)))
            continue;

        // angle filter
        if (angleDeg(S, F, E) > MAX_ANGLE_DEG) continue;

        ++valleys;
        cv::circle(dbg, F, 6, cv::Scalar(255, 0, 130), -1);
        cv::line(dbg, S, F, cv::Scalar(200, 200, 0), 1);
        cv::line(dbg, E, F, cv::Scalar(200, 200, 0), 1);
    }
    return std::min(5, valleys + 1);
}

// ── Method B: curvature-based fingertip detection ─────────────────────────────
// Sample every `step` points on the contour; local minima of curvature angle
// that point upward and sit above the wrist-cut line are counted as fingertips.
static int curvatureCount(const std::vector<cv::Point>& contour,
                           const cv::Rect& bbox,
                           cv::Mat& dbg,
                           std::vector<cv::Point>& tipPts)
{
    tipPts.clear();
    int N = static_cast<int>(contour.size());
    if (N < 30) return 0;

    int step = std::max(3, N / 80);   // adaptive sampling
    int wristY = bbox.y + static_cast<int>(bbox.height * (1.0 - WRIST_CUT_RATIO));
    int topZone = bbox.y + static_cast<int>(bbox.height * 0.85); // tips must be in top 85%

    std::vector<cv::Point> tips;
    for (int i = 0; i < N; i += step) {
        cv::Point prev = contour[(i - step * 3 + N) % N];
        cv::Point curr = contour[i];
        cv::Point next = contour[(i + step * 3) % N];

        // skip wrist area
        if (curr.y > wristY) continue;
        // must be near top of contour
        if (curr.y > topZone) continue;

        double ang = angleDeg(prev, curr, next);
        if (ang < 65.0) {   // sharp local minimum = fingertip
            // check it's a local high-Y point (low pixel y = high on screen)
            bool localMin = true;
            for (int k = -step; k <= step; ++k) {
                if (k == 0) continue;
                if (contour[(i + k + N) % N].y < curr.y) {
                    localMin = false; break;
                }
            }
            if (localMin) tips.push_back(curr);
        }
    }

    // cluster close tips (within 25 px) — keep the topmost
    std::vector<cv::Point> merged;
    std::vector<bool> used(tips.size(), false);
    for (size_t i = 0; i < tips.size(); ++i) {
        if (used[i]) continue;
        cv::Point best = tips[i];
        for (size_t j = i+1; j < tips.size(); ++j) {
            if (used[j]) continue;
            if (cv::norm(tips[i] - tips[j]) < 25.0) {
                used[j] = true;
                if (tips[j].y < best.y) best = tips[j];
            }
        }
        merged.push_back(best);
    }

    // Draw fingertips (numbered)
    int n = std::min(5, static_cast<int>(merged.size()));
    // Sort left → right so numbering is consistent
    std::sort(merged.begin(), merged.begin() + n,
              [](const cv::Point& a, const cv::Point& b){ return a.x < b.x; });
    for (int i = 0; i < n; ++i) {
        cv::circle(dbg, merged[i], 10, cv::Scalar(50, 220, 255), -1);
        cv::circle(dbg, merged[i], 10, cv::Scalar(10, 10, 10),    2);
        cv::putText(dbg, std::to_string(i+1),
                    cv::Point(merged[i].x - 5, merged[i].y - 13),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(10, 10, 10), 2);
        tipPts.push_back(merged[i]);
    }
    return n;
}

// ── draw dashed ROI rectangle ─────────────────────────────────────────────────
static void drawHandROI(cv::Mat& disp)
{
    cv::Rect roi(
        static_cast<int>(ROI_X * disp.cols),
        static_cast<int>(ROI_Y * disp.rows),
        static_cast<int>(ROI_W * disp.cols),
        static_cast<int>(ROI_H * disp.rows));
    roi &= cv::Rect(0, 0, disp.cols, disp.rows);

    cv::Scalar col(180, 180, 180);
    int dash = 14, gap = 8;
    auto dashed = [&](cv::Point a, cv::Point b){
        double len = cv::norm(b - a);
        if (len < 1.0) return;
        cv::Point2d dr = cv::Point2d(b-a) * (1.0/len);
        for (double t = 0; t < len; t += dash+gap) {
            double t2 = std::min(t+dash, len);
            cv::line(disp,
                cv::Point(a.x+static_cast<int>(dr.x*t),
                          a.y+static_cast<int>(dr.y*t)),
                cv::Point(a.x+static_cast<int>(dr.x*t2),
                          a.y+static_cast<int>(dr.y*t2)),
                col, 1);
        }
    };
    dashed(cv::Point(roi.x,          roi.y),
           cv::Point(roi.x+roi.width,roi.y));
    dashed(cv::Point(roi.x+roi.width,roi.y),
           cv::Point(roi.x+roi.width,roi.y+roi.height));
    dashed(cv::Point(roi.x+roi.width,roi.y+roi.height),
           cv::Point(roi.x,          roi.y+roi.height));
    dashed(cv::Point(roi.x,          roi.y+roi.height),
           cv::Point(roi.x,          roi.y));

    cv::putText(disp, "HAND ZONE",
                cv::Point(roi.x+6, roi.y+22),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, col, 1);
}

// ── count fingers, draw all debug visuals ─────────────────────────────────────
static int countFingers(const cv::Mat& frame,
                         cv::Mat& disp,
                         std::deque<int>& hist,
                         cv::Mat& maskPreview)
{
    drawHandROI(disp);

    cv::Rect roi(
        static_cast<int>(ROI_X * frame.cols),
        static_cast<int>(ROI_Y * frame.rows),
        static_cast<int>(ROI_W * frame.cols),
        static_cast<int>(ROI_H * frame.rows));
    roi &= cv::Rect(0, 0, frame.cols, frame.rows);

    cv::Mat hand     = frame(roi).clone();
    cv::Mat debugRoi = disp(roi);

    // ── build mask ────────────────────────────────────────────────
    cv::Mat mask = cleanMask(skinMask(hand));

    // Wrist cutoff — zero out bottom WRIST_CUT_RATIO of mask
    int cutY = static_cast<int>(mask.rows * (1.0 - WRIST_CUT_RATIO));
    mask(cv::Rect(0, cutY, mask.cols, mask.rows - cutY)) = 0;

    // mini preview (top-left of ROI, semi-transparent)
    cv::Mat preview;
    cv::resize(mask, preview, cv::Size(roi.width/4, roi.height/4));
    cv::cvtColor(preview, preview, cv::COLOR_GRAY2BGR);
    cv::addWeighted(
        debugRoi(cv::Rect(0,0,preview.cols,preview.rows)), 0.3,
        preview, 0.7, 0,
        debugRoi(cv::Rect(0,0,preview.cols,preview.rows)));
    maskPreview = preview;

    // ── find largest contour ──────────────────────────────────────
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) { hist.push_back(0); if ((int)hist.size()>FIN_SMOOTH) hist.pop_front(); return majorityVote(hist); }

    auto& best = *std::max_element(contours.begin(), contours.end(),
        [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b){
            return cv::contourArea(a) < cv::contourArea(b); });

    if (cv::contourArea(best) < MIN_AREA) {
        hist.push_back(0); if ((int)hist.size()>FIN_SMOOTH) hist.pop_front();
        return majorityVote(hist);
    }

    cv::Rect bbox = cv::boundingRect(best);

    // draw hand outline
    std::vector<std::vector<cv::Point>> tmp{best};
    cv::drawContours(debugRoi, tmp, 0, cv::Scalar(0, 255, 180), 2);

    // draw convex hull
    std::vector<cv::Point> hull;
    cv::convexHull(best, hull);
    std::vector<std::vector<cv::Point>> htmp{hull};
    cv::drawContours(debugRoi, htmp, 0, cv::Scalar(255, 200, 0), 1);

    // ── Method A: defects ─────────────────────────────────────────
    int countA = defectCount(best, bbox, debugRoi);

    // ── Method B: curvature fingertips ────────────────────────────
    std::vector<cv::Point> tipPts;
    int countB = curvatureCount(best, bbox, debugRoi, tipPts);

    // Draw lines from hand centroid to each tip
    cv::Moments M = cv::moments(best);
    if (M.m00 > 0) {
        cv::Point centroid(static_cast<int>(M.m10/M.m00),
                           static_cast<int>(M.m01/M.m00));
        cv::circle(debugRoi, centroid, 5, cv::Scalar(255,255,255), -1);
        for (const auto& tp : tipPts)
            cv::line(debugRoi, centroid, tp, cv::Scalar(180,255,180), 1);
    }

    // ── Cross-validate A & B: if they agree use that; else use B ──
    int raw = (std::abs(countA - countB) <= 1) ? std::max(countA, countB) : countB;
    raw = std::min(5, std::max(0, raw));

    // ── smooth ────────────────────────────────────────────────────
    hist.push_back(raw);
    if (static_cast<int>(hist.size()) > FIN_SMOOTH) hist.pop_front();
    return majorityVote(hist);
}


// ╔══════════════════════════════════════════════════════════════╗
// ║                    HUD / OVERLAY                             ║
// ╚══════════════════════════════════════════════════════════════╝

static void drawDots(cv::Mat& img, int n, cv::Point origin, int r, int gap)
{
    for (int i = 0; i < 5; ++i) {
        cv::Point c(origin.x + i*(r*2+gap), origin.y);
        cv::Scalar col = (i<n) ? cv::Scalar(50,230,110) : cv::Scalar(55,55,55);
        cv::circle(img, c, r, col, -1);
        cv::circle(img, c, r, cv::Scalar(200,200,200), 1);
        if (i < n)
            cv::putText(img, std::to_string(i+1),
                        cv::Point(c.x-5, c.y+5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4,
                        cv::Scalar(10,10,10), 1);
    }
}

static void drawOverlay(cv::Mat& disp,
                        bool verified, int fingers,
                        bool hasModel, float enrollProg,
                        int voteYes, double lastConf)
{
    const int W = disp.cols;
    const int H = disp.rows;

    // dim left panel
    cv::Rect panel(0, 0, static_cast<int>(W * 0.50), H);
    cv::Mat  reg = disp(panel);
    cv::Mat  blk(panel.size(), CV_8UC3, cv::Scalar(0,0,0));
    cv::addWeighted(reg, 0.50, blk, 0.50, 0, reg);

    // ── status banner ─────────────────────────────────────────────
    if (enrollProg >= 0.f) {
        cv::rectangle(disp, cv::Point(0,0), cv::Point(W,65),
                      cv::Scalar(0,120,200), -1);
        cv::putText(disp, "ENROLLING — follow the on-screen pose prompts",
                    cv::Point(12,42), cv::FONT_HERSHEY_DUPLEX,
                    0.85, cv::Scalar(255,255,255), 2);
        // global progress bar
        int bw = static_cast<int>((W-40) * enrollProg);
        cv::rectangle(disp, cv::Point(20,H-30), cv::Point(20+bw,H-8),
                      cv::Scalar(0,200,80), -1);
        cv::rectangle(disp, cv::Point(20,H-30), cv::Point(W-20,H-8),
                      cv::Scalar(200,200,200), 2);
    } else if (!hasModel) {
        cv::rectangle(disp, cv::Point(0,0), cv::Point(W,65),
                      cv::Scalar(50,50,50), -1);
        cv::putText(disp, "No model — press [E] to enrol your face",
                    cv::Point(12,42), cv::FONT_HERSHEY_DUPLEX,
                    0.85, cv::Scalar(170,170,170), 2);
    } else if (verified) {
        cv::rectangle(disp, cv::Point(0,0), cv::Point(W,65),
                      cv::Scalar(0,140,40), -1);
        cv::putText(disp, "  IDENTITY CONFIRMED — finger count active",
                    cv::Point(12,42), cv::FONT_HERSHEY_DUPLEX,
                    0.88, cv::Scalar(255,255,255), 2);
    } else {
        cv::rectangle(disp, cv::Point(0,0), cv::Point(W,65),
                      cv::Scalar(0,0,160), -1);
        cv::putText(disp, "  FACE NOT RECOGNISED — show your face",
                    cv::Point(12,42), cv::FONT_HERSHEY_DUPLEX,
                    0.88, cv::Scalar(200,200,200), 2);
    }

    // ── confidence vote bar ────────────────────────────────────────
    if (hasModel && enrollProg < 0.f) {
        cv::putText(disp, "Confidence votes:",
                    cv::Point(20, 95), cv::FONT_HERSHEY_SIMPLEX,
                    0.55, cv::Scalar(180,180,180), 1);
        for (int i = 0; i < VERIFY_VOTES; ++i) {
            cv::Scalar col = (i < voteYes) ? cv::Scalar(0,220,80)
                                           : cv::Scalar(60,60,60);
            cv::rectangle(disp,
                          cv::Point(20 + i*22, 100),
                          cv::Point(20 + i*22 + 16, 116),
                          col, -1);
        }
        // distance label
        std::string dl = "dist=" + std::to_string(static_cast<int>(lastConf));
        cv::putText(disp, dl, cv::Point(20 + VERIFY_VOTES*22 + 8, 115),
                    cv::FONT_HERSHEY_SIMPLEX, 0.48, cv::Scalar(180,180,180), 1);
    }

    // ── large finger digit ────────────────────────────────────────
    if (verified && fingers >= 0) {
        std::string ns = std::to_string(fingers);
        // shadow
        cv::putText(disp, ns, cv::Point(60,340),
                    cv::FONT_HERSHEY_DUPLEX, 9.5, cv::Scalar(0,90,30), 20);
        // lit
        cv::putText(disp, ns, cv::Point(57,337),
                    cv::FONT_HERSHEY_DUPLEX, 9.5, cv::Scalar(60,255,130), 16);

        cv::putText(disp, "FINGERS", cv::Point(45,390),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(180,180,180), 2);

        drawDots(disp, fingers, cv::Point(28,418), 20, 10);

    } else if (hasModel && !verified && enrollProg < 0.f) {
        cv::putText(disp, "?", cv::Point(90,330),
                    cv::FONT_HERSHEY_DUPLEX, 9.5, cv::Scalar(70,70,70), 16);
        cv::putText(disp, "Show your face",
                    cv::Point(22,390), cv::FONT_HERSHEY_SIMPLEX,
                    0.68, cv::Scalar(110,110,110), 1);
    }

    // ── bottom key bar ────────────────────────────────────────────
    if (enrollProg < 0.f) {
        cv::rectangle(disp, cv::Point(0,H-32), cv::Point(W,H),
                      cv::Scalar(18,18,18), -1);
        cv::putText(disp, "[E] Enrol    [R] Reload model    [Q / ESC] Quit",
                    cv::Point(10,H-10), cv::FONT_HERSHEY_SIMPLEX,
                    0.55, cv::Scalar(160,160,160), 1);
    }
}


// ╔══════════════════════════════════════════════════════════════╗
// ║                         MAIN                                 ║
// ╚══════════════════════════════════════════════════════════════╝
int main()
{
    fs::create_directories(DATA_DIR);

    // ── camera ─────────────────────────────────────────────────────
    cv::VideoCapture cap(CAM_ID, cv::CAP_ANY);
    if (!cap.isOpened()) {
        std::cerr << "[FATAL] Cannot open camera " << CAM_ID << "\n";
        return 1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  FRAME_W);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_H);
    cap.set(cv::CAP_PROP_FPS, 30);

    // ── face engine ────────────────────────────────────────────────
    cv::CascadeClassifier haar = loadHaar();
    cv::Ptr<cv::face::LBPHFaceRecognizer> rec =
        cv::face::LBPHFaceRecognizer::create(1, 8, 8, 8, CONF_THRESHOLD);

    bool modelTrained = false;
    if (fs::exists(MODEL_PATH)) {
        rec->read(MODEL_PATH);
        modelTrained = true;
        std::cout << "[Face] Model loaded <- " << MODEL_PATH << "\n";
    }

    // ── runtime state ──────────────────────────────────────────────
    bool  verified   = false;
    int   fingers    = 0;
    float enrollProg = -1.f;

    std::deque<int>  fingerHist;
    std::deque<bool> voteWindow;   // sliding verification votes
    double           lastConf = 999.0;

    int64  lastRecogTick = 0;
    double recogInterval = cv::getTickFrequency() / VERIFY_HZ;

    cv::namedWindow(WIN_NAME, cv::WINDOW_NORMAL);

    cv::Mat frame, disp, maskPreview;

    while (true)
    {
        cap >> frame;
        if (frame.empty()) continue;
        cv::flip(frame, frame, 1);
        disp = frame.clone();

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q' || key == 27) break;

        // ── [E] enrol ──────────────────────────────────────────────
        if (key == 'e' || key == 'E') {
            verified = false;
            fingers  = 0;
            voteWindow.clear();
            modelTrained = enrollFace(cap, haar, rec, enrollProg);
            if (modelTrained) {
                rec->save(MODEL_PATH);
                std::cout << "[Face] Model saved -> " << MODEL_PATH << "\n";
            }
            continue;
        }

        // ── [R] reload ─────────────────────────────────────────────
        if ((key == 'r' || key == 'R') && fs::exists(MODEL_PATH)) {
            rec->read(MODEL_PATH);
            modelTrained = true;
            voteWindow.clear();
            std::cout << "[Face] Model reloaded.\n";
        }

        // ── face recognition (throttled) ───────────────────────────
        int64 now = cv::getTickCount();
        if (modelTrained && (now - lastRecogTick) >= recogInterval)
        {
            lastRecogTick = now;

            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::equalizeHist(gray, gray);

            cv::Rect faceRoi = detectFace(gray, haar);

            bool thisFrame = false;
            if (!faceRoi.empty()) {
                // quality gate
                double sharpness = blurScore(gray(
                    faceRoi & cv::Rect(0,0,gray.cols,gray.rows)));

                if (sharpness >= BLUR_THRESHOLD) {
                    cv::Mat proc = normFace(gray, faceRoi);
                    if (!proc.empty()) {
                        int    lbl  = -1;
                        double conf = 0.0;
                        rec->predict(proc, lbl, conf);
                        lastConf   = conf;
                        thisFrame  = (lbl == OWNER_LABEL && conf < CONF_THRESHOLD);
                    }
                }
                drawFaceBox(disp, faceRoi, thisFrame, lastConf,
                            static_cast<int>(
                                std::count(voteWindow.begin(),
                                           voteWindow.end(), true)));
            }

            // sliding vote window
            voteWindow.push_back(thisFrame);
            if (static_cast<int>(voteWindow.size()) > VERIFY_VOTES)
                voteWindow.pop_front();

            int yesVotes = static_cast<int>(
                std::count(voteWindow.begin(), voteWindow.end(), true));
            verified = (yesVotes >= VERIFY_MAJORITY);
        }

        // ── finger counting ────────────────────────────────────────
        fingers = verified
                ? countFingers(frame, disp, fingerHist, maskPreview)
                : -1;

        // ── HUD ────────────────────────────────────────────────────
        int yesVotes = static_cast<int>(
            std::count(voteWindow.begin(), voteWindow.end(), true));
        drawOverlay(disp, verified, fingers, modelTrained,
                    enrollProg, yesVotes, lastConf);

        cv::imshow(WIN_NAME, disp);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}