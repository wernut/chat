use std::env::consts::OS;

fn main() {
    if OS == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
