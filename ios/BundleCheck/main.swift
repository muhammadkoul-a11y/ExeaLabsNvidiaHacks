// Mac-side driver for tests/test_ios_bundle.py: compiled with ../ArrivalCapture/ArrivalCapture/FrameBundle.swift and
// ArrivalAPI.swift, it feeds FrameBundleWriter the frames described in a JSON spec and zips the bundle with ZipWriter,
// or writes the upload body the app would send.
//
//     swiftc -O ios/BundleCheck/main.swift ios/ArrivalCapture/ArrivalCapture/FrameBundle.swift -o bundle_check
//     ./bundle_check spec.json bundle_dir frames.zip
//     ./bundle_check multipart BOUNDARY scan.json frames.zip body.bin   # ArrivalAPI.writeMultipart, as the app uploads
//
// spec: {"room_id", "frames": [{"timestamp", "rgb_jpeg", "rgb_size": [w, h], "depth_f32", "conf_u8" | null,
//        "depth_size": [w, h], "intrinsics": {"fx", "fy", "cx", "cy"}, "camera_to_world": [16, row-major]}]}
// rgb_jpeg, depth_f32 (raw little-endian float32) and conf_u8 (raw bytes) are paths; a non-finite number is the
// string "nan". Prints the kept-frame count.
import Foundation

struct Spec: Decodable {
    struct Frame: Decodable {
        let timestamp: Double
        let rgb_jpeg: String
        let rgb_size: [Int]
        let depth_f32: String
        let conf_u8: String?
        let depth_size: [Int]
        let intrinsics: [String: Double]
        let camera_to_world: [Double]
    }
    let room_id: String
    let frames: [Frame]
}

let args = CommandLine.arguments
if args.count == 4, args[1] == "zip" {
    do {
        try ZipWriter.zip(folder: URL(fileURLWithPath: args[2]), to: URL(fileURLWithPath: args[3]))
        exit(0)
    } catch {
        FileHandle.standardError.write(Data("bundle_check: \(error.localizedDescription)\n".utf8))
        exit(1)
    }
}
if args.count == 6, args[1] == "multipart" {
    do {
        try ArrivalAPI.writeMultipart([
            .init(field: "scan", filename: "kitchen_a.json", contentType: "application/json", file: URL(fileURLWithPath: args[3])),
            .init(field: "frames", filename: "frames.zip", contentType: "application/zip", file: URL(fileURLWithPath: args[4])),
        ], boundary: args[2], to: URL(fileURLWithPath: args[5]))
        exit(0)
    } catch {
        FileHandle.standardError.write(Data("bundle_check: \(error.localizedDescription)\n".utf8))
        exit(1)
    }
}
guard args.count == 4 else {
    FileHandle.standardError.write(Data("usage: bundle_check spec.json bundle_dir frames.zip\n".utf8))
    exit(2)
}
do {
    let decoder = JSONDecoder()  // JSON has no NaN: the spec writes a non-finite number as the string "nan"
    decoder.nonConformingFloatDecodingStrategy = .convertFromString(positiveInfinity: "inf", negativeInfinity: "-inf", nan: "nan")
    let spec = try decoder.decode(Spec.self, from: Data(contentsOf: URL(fileURLWithPath: args[1])))
    let writer = try FrameBundleWriter(root: URL(fileURLWithPath: args[2], isDirectory: true), roomID: spec.room_id)
    for f in spec.frames {
        let raw = try Data(contentsOf: URL(fileURLWithPath: f.depth_f32))
        let depth = raw.withUnsafeBytes { buf in
            (0..<raw.count / 4).map { Float(bitPattern: UInt32(littleEndian: buf.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
        let conf = try f.conf_u8.map { [UInt8](try Data(contentsOf: URL(fileURLWithPath: $0))) }
        let k = FrameBundleWriter.Intrinsics(fx: f.intrinsics["fx"]!, fy: f.intrinsics["fy"]!,
                                             cx: f.intrinsics["cx"]!, cy: f.intrinsics["cy"]!)
        try writer.add(timestamp: f.timestamp, rgbJPEG: Data(contentsOf: URL(fileURLWithPath: f.rgb_jpeg)),
                       rgbWidth: f.rgb_size[0], rgbHeight: f.rgb_size[1], depth: depth,
                       depthWidth: f.depth_size[0], depthHeight: f.depth_size[1], confidence: conf,
                       intrinsics: k, cameraToWorld: f.camera_to_world)
    }
    try writer.finish()
    try ZipWriter.zip(folder: writer.root, to: URL(fileURLWithPath: args[3]))
    print(writer.count)
} catch {
    FileHandle.standardError.write(Data("bundle_check: \(error.localizedDescription)\n".utf8))
    exit(1)
}
