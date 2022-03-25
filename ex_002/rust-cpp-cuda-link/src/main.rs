
#[link(name = "cuda_wrapper")]
extern "C" {
    fn cuda_wrapper();
}

fn main() {
    println!("Hello, world!");
    unsafe { cuda_wrapper(); }
}
