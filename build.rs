extern crate cc;

#[cfg(not(feature = "gpu"))]
fn main() {
    let files = [
        "bindings/bindings.cpp",
        "bindings/GameBinding.cpp",
        "Game/Game.cpp",
        "NeuralNetwork/NN.cpp",
        "NeuralNetwork/Topology.cpp",
        "Private/Generation.cpp",
        "Private/Mutation.cpp",
        "Private/MutationField.cpp",
        "Private/Gene.cpp",
        "Private/Random.cpp",
        "Private/Species.cpp",
        "Serializer/Serializable.cpp",
        "Train/Train.cpp",
        "Timer.cpp",
        "TopologyParser/TopologyParser.cpp"
    ];

    let mut builder = cc::Build::new();
    builder.include(".")
        .include("./bindings")
        .include("./Game")
        .include("./NeuralNetwork")
        .include("./Private")
        .include("./Serializer")
        .include("./Threading")
        .include("./Train")
        .include("./TopologyParser");
    files.iter().for_each(|file| { builder.file(file); });
    let target = std::env::var("TARGET").unwrap();

    builder
        .cpp(true)
        .cpp_link_stdlib("stdc++")
        .warnings_into_errors(true);

    if !target.contains("wasm32") {
        builder.define("__MULTITHREADED__", "1")
    }

    if cfg!(debug_assertions) {
        builder
            .flag("-Og")
            .flag("-g");
    } else if !target.contains("wasm32") {
        builder.flag("-Ofast")
            .flag("-march=native")
            .flag("-ffast-math")
            .flag("-frename-registers")
            .flag("-flto")
            .flag("-fwhole-program");
    }

    builder.compile("neat");

    files.iter().for_each(|file| {
        println!("cargo:rerun-if-changed={}", file);
    });
    println!("cargo:rerun-if-changed=build.rs");
}

#[cfg(feature = "gpu")]
fn main() {
    if let Ok(cuda_path) = std::env::var("CUDA_HOME") {
        println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
    } else {
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    }
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=curand");
    println!("cargo:rustc-link-lib=dylib=cudadevrt");

    let files = [
        "NeuralNetwork/NN.cpp",
        "bindings/bindings.cpp",
        "bindings/GameBinding.cpp",
        "Game/Game.cpp",
        "NeuralNetwork/Topology.cpp",
        "Private/Generation.cpp",
        "Private/Mutation.cpp",
        "Private/MutationField.cpp",
        "Private/Gene.cpp",
        "Private/Random.cpp",
        "Private/Species.cpp",
        "Serializer/Serializable.cpp",
        "Train/Train.cpp",
        "Timer.cpp",
        "TopologyParser/TopologyParser.cpp"
    ];

    let mut builder = cc::Build::new();
    builder.include(".")
        .include("./bindings")
        .include("./Game")
        .include("./NeuralNetwork")
        .include("./Private")
        .include("./Serializer")
        .include("./Threading")
        .include("./Train")
        .include("./TopologyParser");
    files.iter().for_each(|file| { builder.file(file); });
    builder
        .cpp(true)
        .cpp_link_stdlib("stdc++")
        .define("CUDA_ENABLED", "1")
        .define("__MULTITHREADED__", "1")
        .warnings_into_errors(true);

    if cfg!(debug_assertions) {
        builder
            .flag("-Og")
            .flag("-g");
    } else {
        builder.flag("-Ofast")
            .flag("-march=native")
            .flag("-ffast-math")
            .flag("-frename-registers")
            .flag("-flto")
            .flag("-fwhole-program");
    }
    builder.compile("neat");

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=curand");
    println!("cargo:rustc-link-lib=dylib=cudadevrt");

    let cuda_file = "GPU/NN.cu";
    cc::Build::new().cuda(true)
        .no_default_flags(true)
        .flag("-gencode").flag("arch=compute_52,code=sm_52") // Generate code for Maxwell (GTX 970, 980, 980 Ti, Titan X).
        .flag("-gencode").flag("arch=compute_53,code=sm_53") // Generate code for Maxwell (Jetson TX1).
        .flag("-gencode").flag("arch=compute_61,code=sm_61") // Generate code for Pascal (GTX 1070, 1080, 1080 Ti, Titan Xp).
        .flag("-gencode").flag("arch=compute_60,code=sm_60") // Generate code for Pascal (Tesla P100).
        .flag("-gencode").flag("arch=compute_62,code=sm_62") // Generate code for Pascal (Jetson TX2).
        .static_flag(true)
        .file(cuda_file).compile("GPU");

    println!("cargo:rustc-link-lib=dylib=cuda");
    println!("cargo:rustc-link-lib=dylib=cudart");

    files.iter().for_each(|file| {
        println!("cargo:rerun-if-changed={}", file);
    });
    println!("cargo:rerun-if-changed={}", cuda_file);
    println!("cargo:rerun-if-changed=build.rs");
}