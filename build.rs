extern crate cc;

fn main() {
    cc::Build::new()
        .cpp(true)
        .flag("-std=c++11")
        .flag("-Wno-unused-parameter")
        .file("vendor/libmf/mf.cpp")
        .compile("mf");
}
