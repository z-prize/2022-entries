use fpga::F1;
use msm_fpga::{always_timed, harness_scalars, load_harness_points, timed, App};

fn main() {
    let size = std::env::args()
        .nth(1)
        .expect("pass with SIZE argument")
        .parse()
        .expect("SIZE invalid as u8");

    let name = std::env::args().nth(2).expect("pass with NAME argument");

    let f1 = F1::new(0, 0x500).unwrap();
    let mut app = App::new(f1, size);

    let (beta, points) = load_harness_points(size as _, &name);
    let (scalars, sum) = harness_scalars(&beta, size as _);

    timed("setting points", || app.set_preprocessed_points(&points));

    // the MSM
    let total = always_timed(&format!("size {} MSM", size), || app.msm(&scalars));

    if total != sum {
        println!("\n==> FAILURE <==");
        std::process::exit(1);
    } else {
        println!("\n==> SUCCESS <==");
    }
}
