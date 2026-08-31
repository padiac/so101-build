// =============================================================
// WaveShare_Mounting_Plate_SO101 -- M2.5 COUNTERBORE rework
//
// Problem: the stock plate has four teardrop holes of 5.0 mm dia.
// An M2.5 hex brass standoff is also 5.0 mm across flats, so the
// standoff cannot be pressed in -- zero clearance.
//
// Fix: fill the four stock holes, then cut M2.5 clearance holes
// with a STRAIGHT-WALLED counterbore on the boss side, so an
// ordinary cylindrical-head M2.5 screw (socket cap / pan / cheese)
// sits fully below the surface. A conical countersink would only
// work with a tapered flat-head screw -- that was wrong.
//
//   board + brass standoffs   ->  flat face   (z = 0)
//   arm base / snap-fit boss  ->  boss face   (z = 4)
//   screw head sits inside the counterbore at z = 4
//
// Measured from the original STL (ray-cast cross sections):
//   plate           51.0 x 42.0 x 4.0 mm  (boss up to z = 7.6)
//   hole centres    (+-18.5, +-14.0)  ->  pitch 37.0 x 28.0
//   stock hole      teardrop, dia 5.0 circle + 1.0 mm tip toward -Y
//
// Head heights for reference (pick CB_DEPTH >= head height):
//   M2.5 socket cap  DIN 912   head 4.5 dia x 2.5 tall
//   M2.5 pan head    DIN 7985  head 5.0 dia x 1.8 tall
//   M2.5 cheese head DIN 84    head 4.5 dia x 1.6 tall
// =============================================================

$fn = 120;

STL = "C:/Users/pppad/AppData/Local/Temp/claude/E--Repo/29a8a352-8fae-400c-b0c1-f629c092a863/scratchpad/ws_plate.stl";

PLATE_T   = 4.0;    // thickness of the flat plate (hole depth)
HOLE_X    = 18.5;   // hole centre X offset
HOLE_Y    = 14.0;   // hole centre Y offset

M25_CLEAR = 2.8;    // M2.5 clearance hole (screw shank passes freely)

CB_DIA    = 5.4;    // counterbore dia -- clears a 5.0 head with margin
CB_DEPTH  = 2.6;    // counterbore depth; set >= your screw head height
                    //   2.0 -> pan / cheese head
                    //   2.6 -> socket cap head (default, fits all)

FILL_DIA  = 5.3;    // slight oversize so the boolean union is clean
FILL_TIP  = 3.9;    // teardrop tip reach, toward -Y

CENTRES = [
    [-HOLE_X, -HOLE_Y],
    [ HOLE_X, -HOLE_Y],
    [-HOLE_X,  HOLE_Y],
    [ HOLE_X,  HOLE_Y],
];

// Stock teardrop cross-section, slightly oversized.
module teardrop_2d() {
    hull() {
        circle(d = FILL_DIA);
        translate([0, -FILL_TIP]) circle(d = 0.01);
    }
}

module plug() {
    linear_extrude(height = PLATE_T) teardrop_2d();
}

// M2.5 through hole + straight counterbore opening toward +Z.
module m25_counterbore() {
    translate([0, 0, -1])
        cylinder(d = M25_CLEAR, h = PLATE_T + 2);
    translate([0, 0, PLATE_T - CB_DEPTH])
        cylinder(d = CB_DIA, h = CB_DEPTH + 1);
}

difference() {
    union() {
        import(STL, convexity = 10);
        for (c = CENTRES) translate([c[0], c[1], 0]) plug();
    }
    for (c = CENTRES) translate([c[0], c[1], 0]) m25_counterbore();
}

echo(str("plate ", PLATE_T, "mm | bore ", M25_CLEAR,
         " | counterbore ", CB_DIA, " x ", CB_DEPTH,
         " | material left under head: ", PLATE_T - CB_DEPTH, "mm"));
