use std::ptr;
use std::time::{SystemTime};
use libc::size_t;

#[link(name = "cuda_wrapper")]
extern "C" {
  fn cuda_wrapper(arr: *mut f64, len: size_t);
}

fn main() {
    println!("Hello, world!");
    let mut arr: Vec<f64> = Vec::with_capacity(100_000_000);
    for i in 0..arr.capacity() {
      arr.push(i as f64);
    }
    let start = SystemTime::now();
    unsafe { cuda_wrapper(arr.as_mut_ptr(), arr.len()); }
    let d = 
    match start.elapsed() {
      Ok(d) => d,
      Err(e) => {
        eprintln!("ERROR: {:?}", e);
        return;
      }
    };

//    for i in 0..arr.len() {
//      print!("{},", arr[i]);
//    }
//    println!("");

    println!("time passed: {} msec", d.as_millis());
}
