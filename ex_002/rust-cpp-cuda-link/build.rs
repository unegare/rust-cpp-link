use std::process::Command;

fn main() {
    Command::new("make")
        .current_dir("./cpp")
        .status()
        .expect("make failed");
    println!("cargo:rustc-link-search=native=./cpp");
    println!("cargo:rustc-link-lib=dylib=cuda_wrapper");
}
