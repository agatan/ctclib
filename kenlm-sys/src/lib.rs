#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(clippy::redundant_static_lifetimes)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::upper_case_acronyms)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[test]
fn test_works() {
use std::ffi::CString;
    unsafe {
        let mut config = lm_ngram_Config::new();
        lm_ngram_Config_Config(&mut config as *mut _);
        let file_path = CString::new("test.arpa").unwrap();
        lm_ngram_LoadVirtual(file_path.as_ptr(), &mut config as *mut _);
    }
}
