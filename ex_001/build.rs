use std::process::Command;

fn main() {
    Command::new("make")
        .current_dir("./cpp")
        .status()
        .expect("failed to make!");
    println!("cargo:rustc-link-search=native=./cpp");
    println!("cargo:rustc-link-lib=dylib=mylib");
}
