use std::ptr;
use libc::size_t;

#[link(name = "cuda_wrapper")]
extern "C" {
  fn cuda_wrapper(arr: *mut [f64], len: size_t);
}

fn main() {
    println!("Hello, world!");
    let mut arr: [f64; 32] = [0f64; 32];
    for i in 0..arr.len() {
      arr[i] = i as f64;
    }
    unsafe { cuda_wrapper(&mut arr, arr.len()); }
    for i in 0..arr.len() {
      print!("{},", arr[i]);
    }
    println!("");
}
