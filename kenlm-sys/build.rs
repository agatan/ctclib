use std::{path::PathBuf, process::Command};

use cmake::Config;

fn main() {
    let target = std::env::var("TARGET").unwrap();
    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let kenlm_root = out_path.join("kenlm");

    if !kenlm_root.exists() {
        // Copy
        Command::new("cp")
            .args(&["-r", "kenlm", kenlm_root.to_str().unwrap()])
            .status()
            .unwrap();
    }

    // CMake
    let dst = Config::new(&kenlm_root).profile("Release").build();

    // bindgen build
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(&["-x", "c++", "-std=c++11"])
        .clang_arg(format!("-I{}", kenlm_root.display()))
        .allowlist_type("lm::base::Model")
        .opaque_type("lm::base::Model")
        .allowlist_type("lm::ngram::State")
        .opaque_type("lm::ngram::State")
        .allowlist_type("lm::base::Vocabulary")
        .opaque_type("lm::base::Vocabulary")
        .opaque_type("lm::ngram::State")
        .allowlist_type("lm::ngram::Config")
        .opaque_type("std::.*")
        .allowlist_function("lm_ngram_LoadVirtual.*")
        .allowlist_function("lm_base_Vocabulary_.*")
        .allowlist_function("lm_base_Model_.*")
        .generate()
        .expect("Unable to generate bindings");
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");
    println!("cargo:rerun-if-changed=wrapper.cpp");
    cc::Build::new().cpp(true).file("wrapper.cpp").flag("-fkeep-inline-functions").compile("wrapper");

    // link to appropriate C++ lib
    if target.contains("apple") {
        println!("cargo:rustc-link-lib=c++");
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }

    println!("cargo:rustc-link-search={}", out_path.join("lib").display());
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=kenlm");
    println!("cargo:rustc-link-lib=static=kenlm_util");
    println!("cargo:rustc-link-lib=static=kenlm_filter");
    println!("cargo:rustc-link-lib=static=kenlm_builder");
    println!("cargo:rustc-link-lib=dylib=lzma");
    println!("cargo:rustc-link-lib=dylib=bz2");
    println!("cargo:rustc-link-lib=dylib=z");
}
