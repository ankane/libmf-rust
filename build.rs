extern crate cc;

fn main() {
    cc::Build::new()
        .cpp(true)
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-Wno-unused-parameter")
        .file("vendor/libmf/mf.cpp")
        .compile("mf");
}
