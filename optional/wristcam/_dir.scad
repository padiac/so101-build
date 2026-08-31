NX = 0.42305; NY = 0.90611;
M = [[ NX,  NY, 0, 0],
     [ NY, -NX, 0, 0],
     [  0,   0, 1, 0]];
CU = -28.53; CV = 17.55;

module mark(du, dv, s) {
    multmatrix(M) translate([-53.3, CU + du, CV + dv]) cube([14, s, s], center = true);
}
import("SO-ARM101_camera_wrist_mount_BIGGER.stl");
// +v 方向放一个大方块，-v 方向放一个小方块
color("red")   mark(0,  16.0, 5);
color("blue")  mark(0, -16.0, 2.5);
