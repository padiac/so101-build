// =====================================================================
// SO-ARM101 顶部相机支架 v3 —— 扩大开口 + 加大板子
//
// v2 的问题：开口还是塞不进 U20CAM-1080P（相机板边缘有芯片顶住），
//            而 v2 已经快挖到板边，没法再扩。
// v3 解法：**把板子本身加大**。用户确认：板子变宽变长没有代价，
//          只要 (a) 角度不变 (b) 4 个螺丝孔 27x27 间距不变。
//
// 于是唯一剩下的硬约束是「开口边缘到螺丝孔的筋」——这个加大板子救不了，
// 所以四角窄处只能给到 21.0（筋宽 1.9mm）。
//
// 实测原件几何（板平面局部坐标系，射线投射切片得到）：
//   板法向   n = (0.423, -0.906, 0)     板厚 3.0mm  (n·p: -318.75 .. -315.75)
//   开口中心 (CU, CV) = (-179.94, 0.04)
//   4 个螺丝孔 相对开口中心 (±13.5, ±13.5)，孔径约 2.2
//
// 装配后的方向对应（已由用户实物确认）：
//   我的 u 轴 = 上下      我的 v 轴 = 左右
// =====================================================================

STL = "cam_mount_top.stl";

// ---- 开口（相对开口中心的半宽）----
UD_HALF     = 16.0;   // 上下最大跨度的一半  -> 32.0
LR_HALF     = 16.0;   // 左右最大跨度的一半  -> 32.0
NARROW_HALF = 10.5;   // 四角窄处的一半      -> 21.0

HOLE_PITCH_HALF = 13.75;  // 螺丝孔中心距的一半 -> 27.5（实测相机板）
HOLE_D          = 2.2;    // 孔径（过 M2 螺丝）

// ---- 补板（相对开口中心的半宽）----
SLAB_U_HALF = 19.5;   // 补板要盖住 32 的开口 + 3.5mm 边距
SLAB_V_HALF = 19.5;
SLAB_R      = 3;      // 圆角

// ---- 实测常量 ----
CU = -179.94;  CV = 0.04;
PLATE_N0 = -318.75;   // 顶部件板厚 3.0mm
PLATE_N1 = -315.75;
CUT_N0   = -322;
CUT_N1   = -312;

NX = 0.42305; NY = -0.90611;
M = [[ NX,  NY, 0, 0],
     [ NY, -NX, 0, 0],
     [  0,   0, 1, 0]];

// 局部坐标里的一个盒子：法向 n0..n1，面内以开口中心为心
module local_box(n0, n1, hu, hv) {
    multmatrix(M)
        translate([(n0 + n1) / 2, CU, CV])
            cube([n1 - n0, hu * 2, hv * 2], center = true);
}

// 圆角补板
module slab() {
    multmatrix(M)
        translate([(PLATE_N0 + PLATE_N1) / 2, CU, CV])
            rotate([0, 90, 0])
                linear_extrude(height = PLATE_N1 - PLATE_N0, center = true)
                    offset(r = SLAB_R)
                        square([2 * (SLAB_V_HALF - SLAB_R),
                                2 * (SLAB_U_HALF - SLAB_R)], center = true);
}

$fn = 48;

difference() {
    union() {
        import(STL, convexity = 10);
        slab();
    }
    union() {
        local_box(CUT_N0, CUT_N1, NARROW_HALF, LR_HALF);   // 左右长条
        local_box(CUT_N0, CUT_N1, UD_HALF, NARROW_HALF);   // 上下长条
    }
    // 补板盖住了原来的 4 个螺丝孔，按【实测的相机孔距】重新打通。
    //
    // 官方支架是 27.0，但实测相机板是 27.5：
    //   游标卡尺内缘到内缘 25.0，孔为 M2.5（孔径 2.5），中心距 = 25.0 + 2.5 = 27.5
    // 几何自洽性校验：孔心 ±13.75，孔外缘 15.0，板半宽 16 -> 离板边 1mm，成立。
    //   （若按孔距 30 算，孔外缘到 17.5，会豁出 32mm 的板外，不可能。）
    //
    // 孔往外挪 0.25 的副作用是好的：到开口的筋 1.90 -> 2.15mm。
    for (su = [-1, 1], sv = [-1, 1])
        multmatrix(M)
            translate([(CUT_N0 + CUT_N1) / 2,
                       CU + su * HOLE_PITCH_HALF,
                       CV + sv * HOLE_PITCH_HALF])
                rotate([0, 90, 0])
                    cylinder(h = CUT_N1 - CUT_N0, d = HOLE_D, center = true);
}
