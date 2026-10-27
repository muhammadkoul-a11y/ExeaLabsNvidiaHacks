import Foundation

/// One saved capture: `Documents/scans/<id>/` holds `roomplan.json` (CapturedRoom via JSONEncoder,
/// the format `arrival/scan/roomplan.py` reads), `room.usdz`, optionally `frames/` (a CAP-2a RGB-D bundle from
/// `FrameRecorder`) and this record as `scan_meta.json`.
struct Scan: Identifiable, Codable, Equatable {
    var id: String
    var roomName: String
    var created: Date
    var jobID: String?
    var jobStatus: String?
    var jobStage: String?
    var jobError: String?
    var uploadedTo: String?
    var frameCount: Int?
    var heldoutCount: Int?
    var framesNote: String?  // why no frames were kept, when recording was on

    var isJobFinished: Bool { jobStatus == "done" || jobStatus == "failed" || jobStatus == "cancelled" }
}

