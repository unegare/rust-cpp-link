//use libc::int_t;

#[link(name = "mylib")]
extern "C" {
    fn add(a: i32, b: i32) -> i32;
}

fn main() {
    println!("Hello, world!");
    println!("add: {}", unsafe { add(5,7) });
}
