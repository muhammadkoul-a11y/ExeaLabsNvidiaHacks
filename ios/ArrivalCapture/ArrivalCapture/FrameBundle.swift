import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Writes a CAP-2a RGB-D frame bundle (`frames_manifest.json` plus the files it names), the format
/// `arrival/scan/rgbd.py` loads and `POST /jobs` accepts zipped in the `frames` field. Foundation, CoreGraphics
/// and ImageIO only, so `tests/test_ios_bundle.py` can compile and run it on a Mac without a device.
///
/// Axes: poses are ARKit camera-to-world matrices (x right, y up, z backward); `camera_to_room` is that pose
/// composed with `arkitWorldToRoom`, the Y-up to Z-up swap. The server aligns this room frame with the twin's
/// using the scan's own `arkit_world_to_room`, so the bundle does not need to know how the walls were squared.
final class FrameBundleWriter {
    struct Intrinsics {
        var fx: Double, fy: Double, cx: Double, cy: Double
    }

    /// ARKit world (Y up) to the bundle's room frame (Z up): (x, y, z) -> (x, -z, y), as `arrival/scan/roomplan.py`.
    static let arkitWorldToRoom: [Double] = [1, 0, 0, 0,
                                             0, 0, -1, 0,
                                             0, 1, 0, 0,
                                             0, 0, 0, 1]

    /// Every `heldoutEvery`-th frame is held out for the realism comparison.
    static let heldoutEvery = 5

    enum BundleError: LocalizedError {
        case sizeChanged(String)
        case badDepth(Int, Int, Int)
        case encode(String)
        case empty

        var errorDescription: String? {
            switch self {
            case .sizeChanged(let what): return "The \(what) resolution changed during the recording."
            case .badDepth(let n, let w, let h): return "Depth has \(n) values, expected \(w)x\(h)."
            case .encode(let what): return "Could not encode \(what)."
            case .empty: return "No frames were recorded."
            }
        }
    }

    let root: URL
    let roomID: String
    private(set) var frames: [[String: Any]] = []
    private var rgbSize: (Int, Int)?
    private var depthSize: (Int, Int)?
    private var lastTimestamp = -Double.infinity

    init(root: URL, roomID: String) throws {
        self.root = root
        self.roomID = roomID
        for sub in ["rgb", "depth", "conf"] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(sub), withIntermediateDirectories: true)
        }
    }

    var count: Int { frames.count }

    /// Adds one frame. `rgbJPEG` is an encoded image of `rgbWidth` x `rgbHeight`; `depth` is metres along the
    /// optical axis, row-major `depthWidth` x `depthHeight` (0 or non-finite = no reading); `confidence` is the
    /// ARKit confidence level per depth pixel (0 low, 1 medium, 2 high), or nil. `cameraToWorld` is a
    /// row-major 4x4. Returns false, writing nothing, for a frame the loader would reject: a timestamp that does
    /// not increase, or a non-finite pose or intrinsic.
    @discardableResult
    func add(timestamp: Double, rgbJPEG: Data, rgbWidth: Int, rgbHeight: Int,
             depth: [Float], depthWidth: Int, depthHeight: Int, confidence: [UInt8]?,
             intrinsics k: Intrinsics, cameraToWorld: [Double]) throws -> Bool {
        // JSONSerialization raises on NaN, and the loader rejects non-finite poses anyway
        guard timestamp > lastTimestamp, timestamp.isFinite, cameraToWorld.count == 16, cameraToWorld.allSatisfy(\.isFinite),
              [k.fx, k.fy, k.cx, k.cy].allSatisfy(\.isFinite) else { return false }
        if let s = rgbSize, s != (rgbWidth, rgbHeight) { throw BundleError.sizeChanged("RGB") }
        if let s = depthSize, s != (depthWidth, depthHeight) { throw BundleError.sizeChanged("depth") }
        guard depth.count == depthWidth * depthHeight else { throw BundleError.badDepth(depth.count, depthWidth, depthHeight) }
        rgbSize = (rgbWidth, rgbHeight)
        depthSize = (depthWidth, depthHeight)

        let id = String(format: "f%05d", frames.count)
        let rgb = "rgb/\(id).jpg", dep = "depth/\(id).npy", conf = "conf/\(id).png"
        try rgbJPEG.write(to: root.appendingPathComponent(rgb), options: .atomic)
        try Self.npy(depth, width: depthWidth, height: depthHeight).write(to: root.appendingPathComponent(dep), options: .atomic)
        var entry: [String: Any] = [
            "id": id, "timestamp": timestamp, "rgb": rgb, "depth": dep,
            "intrinsics": ["fx": k.fx, "fy": k.fy, "cx": k.cx, "cy": k.cy],
            "camera_to_room": Self.multiply(Self.arkitWorldToRoom, cameraToWorld),
            "split": (frames.count + 1) % Self.heldoutEvery == 0 ? "heldout" : "train",
        ]
        if let confidence, confidence.count == depthWidth * depthHeight {
            try Self.grayPNG(confidence, width: depthWidth, height: depthHeight).write(to: root.appendingPathComponent(conf), options: .atomic)
            entry["confidence"] = conf
        }
        frames.append(entry)
        lastTimestamp = timestamp
        return true
    }

    /// Writes `frames_manifest.json`; call once after the last frame.
    func finish() throws {
        guard let rgbSize, let depthSize, !frames.isEmpty else { throw BundleError.empty }
        let manifest: [String: Any] = [
            "schema_version": 1, "room_id": roomID,
            "axes": ["camera": "arkit_x_right_y_up_z_backward", "room": "z_up_metres"],
            "arkit_world_to_room": Self.arkitWorldToRoom,
            "rgb_size": [rgbSize.0, rgbSize.1], "depth_size": [depthSize.0, depthSize.1], "depth_unit": "m",
            "generator": "ArrivalCapture FrameRecorder (CAP-2)",
            "frames": frames,
        ]
        let data = try JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: root.appendingPathComponent("frames_manifest.json"), options: .atomic)
    }

    // MARK: encoders

    /// Row-major 4x4 product a * b.
    static func multiply(_ a: [Double], _ b: [Double]) -> [Double] {
        var out = [Double](repeating: 0, count: 16)
        for r in 0..<4 {
            for c in 0..<4 {
                var sum = 0.0
                for k in 0..<4 { sum += a[r * 4 + k] * b[k * 4 + c] }
                out[r * 4 + c] = sum
            }
        }
        return out
    }

    /// NumPy .npy v1.0, little-endian float32, C order: what `np.load(..., allow_pickle=False)` reads.
    static func npy(_ values: [Float], width: Int, height: Int) -> Data {
        var header = "{'descr': '<f4', 'fortran_order': False, 'shape': (\(height), \(width)), }"
        let unpadded = 10 + header.utf8.count + 1  // magic(6) + version(2) + length(2) + header + "\n"
        header += String(repeating: " ", count: (64 - unpadded % 64) % 64) + "\n"
        var out = Data([0x93] + Array("NUMPY".utf8) + [1, 0])
        let length = UInt16(header.utf8.count)
        out.append(contentsOf: [UInt8(length & 0xff), UInt8(length >> 8)])
        out.append(Data(header.utf8))
        values.withUnsafeBufferPointer { buf in
            for v in buf {
                withUnsafeBytes(of: v.bitPattern.littleEndian) { out.append(contentsOf: $0) }
            }
        }
        return out
    }

    /// 8-bit grayscale PNG.
    static func grayPNG(_ pixels: [UInt8], width: Int, height: Int) throws -> Data {
        guard let provider = CGDataProvider(data: Data(pixels) as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: width,
                                  space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0),
                                  provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { throw BundleError.encode("confidence image") }
        let out = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(out, UTType.png.identifier as CFString, 1, nil) else {
            throw BundleError.encode("confidence PNG")
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw BundleError.encode("confidence PNG") }
        return out as Data
    }
}

/// A store-only (uncompressed) zip of a folder, with the folder's contents at the archive root: the layout
/// `POST /jobs` wants for `frames` (`frames_manifest.json` at the root). JPEG and PNG are already compressed, so
/// storing costs little. No zip64: the server's own caps (1 GB, 4000 files) are far below its limits.
enum ZipWriter {
    enum ZipError: LocalizedError {
        case tooLarge
        var errorDescription: String? { "The frames bundle is too large to zip (over 4 GB or 65535 files)." }
    }

    static func zip(folder: URL, to output: URL) throws {
        let fm = FileManager.default
        let base = folder.standardizedFileURL.path
        var files: [(name: String, url: URL)] = []
        if let walker = fm.enumerator(at: folder, includingPropertiesForKeys: [.isRegularFileKey]) {
            for case let url as URL in walker where (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true {
                let path = url.standardizedFileURL.path
                files.append((String(path.dropFirst(base.count + 1)), url))
            }
        }
        files.sort { $0.name < $1.name }
        guard files.count < 0xffff else { throw ZipError.tooLarge }

        // Publish only a complete archive; a failed read/write must leave the previous zip intact.
        let staging = output.deletingLastPathComponent()
            .appendingPathComponent(".\(output.lastPathComponent).\(UUID().uuidString).tmp")
        defer { try? fm.removeItem(at: staging) }
        fm.createFile(atPath: staging.path, contents: nil)
        let handle = try FileHandle(forWritingTo: staging)
        defer { try? handle.close() }
        var central = Data()
        var offset: UInt64 = 0
        for file in files {
            let data = try Data(contentsOf: file.url)
            let name = Data(file.name.utf8)
            guard data.count < 0xffff_ffff, offset < 0xffff_ffff else { throw ZipError.tooLarge }
            let crc = crc32(data)
            var local = Data()
            local.append(le32(0x0403_4b50))
            local.append(le16(20)); local.append(le16(0x0800)); local.append(le16(0))  // version, UTF-8 names, stored
            local.append(le16(0)); local.append(le16(0x21))  // 00:00, 1980-01-01
            local.append(le32(crc)); local.append(le32(UInt32(data.count))); local.append(le32(UInt32(data.count)))
            local.append(le16(UInt16(name.count))); local.append(le16(0))
            local.append(name)
            try handle.write(contentsOf: local)
            try handle.write(contentsOf: data)

            central.append(le32(0x0201_4b50))
            central.append(le16(20)); central.append(le16(20)); central.append(le16(0x0800)); central.append(le16(0))
            central.append(le16(0)); central.append(le16(0x21))
            central.append(le32(crc)); central.append(le32(UInt32(data.count))); central.append(le32(UInt32(data.count)))
            central.append(le16(UInt16(name.count))); central.append(le16(0)); central.append(le16(0))
            central.append(le16(0)); central.append(le16(0)); central.append(le32(0o100644 << 16))
            central.append(le32(UInt32(offset)))
            central.append(name)
            offset += UInt64(local.count + data.count)
        }
        guard offset < 0xffff_ffff else { throw ZipError.tooLarge }
        var end = Data()
        end.append(le32(0x0605_4b50)); end.append(le16(0)); end.append(le16(0))
        end.append(le16(UInt16(files.count))); end.append(le16(UInt16(files.count)))
        end.append(le32(UInt32(central.count))); end.append(le32(UInt32(offset))); end.append(le16(0))
        try handle.write(contentsOf: central)
        try handle.write(contentsOf: end)
        try handle.close()
        if fm.fileExists(atPath: output.path) {
            _ = try fm.replaceItemAt(output, withItemAt: staging)
        } else {
            try fm.moveItem(at: staging, to: output)
        }
    }

    private static let table: [UInt32] = (0..<256).map { n -> UInt32 in
        var c = UInt32(n)
        for _ in 0..<8 { c = c & 1 == 1 ? 0xedb8_8320 ^ (c >> 1) : c >> 1 }
        return c
    }

    static func crc32(_ data: Data) -> UInt32 {
        var c: UInt32 = 0xffff_ffff
        data.withUnsafeBytes { raw in
            for b in raw.bindMemory(to: UInt8.self) { c = table[Int((c ^ UInt32(b)) & 0xff)] ^ (c >> 8) }
        }
        return c ^ 0xffff_ffff
    }

    private static func le16(_ v: UInt16) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
    private static func le32(_ v: UInt32) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
}
