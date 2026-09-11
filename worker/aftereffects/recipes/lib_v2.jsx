/*
 * Helpers used by text_overlay_v2 (and by inspect_project.jsx when it inspects
 * a v2 project). Kept OUT of lib.jsx so text_overlay_v1's recipe revision
 * (sha over lib.jsx + its own scripts) stays at the pin the platform holds.
 * Evaluated after lib.jsx; it overrides AE.layerSummary / AE.compSummary with
 * richer versions (stretch, stroke, effects, mattes, masks, key counts,
 * precomp inner layer, comp motion blur).
 *
 * ExtendScript notes that bit while writing v2: nested ternaries parse as
 * (a ? b : c) ? d : e - never nest them; reading TextDocument.fillColor
 * throws "invalid numeric result"; horizontalScale/verticalScale are factors.
 */

AE.justification = function (just) {
    if (just === "left") { return ParagraphJustification.LEFT_JUSTIFY; }
    if (just === "right") { return ParagraphJustification.RIGHT_JUSTIFY; }
    return ParagraphJustification.CENTER_JUSTIFY;
};

AE.findLayer = function (comp, name) {
    for (var i = 1; i <= comp.numLayers; i++) { if (comp.layer(i).name === name) { return comp.layer(i); } }
    return null;
};

AE.keyCounts = function (layer) {
    var out = {};
    var names = { position: "Position", scale: "Scale", rotation: "Rotation", opacity: "Opacity" };
    for (var k in names) {
        if (!names.hasOwnProperty(k)) { continue; }
        try { var p = layer.property(names[k]); if (p && p.numKeys > 0) { out[k] = p.numKeys; } } catch (e) { }
    }
    try {
        var st = layer.property("Source Text");
        if (st && st.numKeys > 0) { out.stretch = st.numKeys; }
    } catch (e2) { }
    try {
        var parade = layer.property("ADBE Effect Parade");
        for (var i = 1; i <= parade.numProperties; i++) {
            var fx = parade.property(i);
            if (fx.name === "brightness" && fx.matchName === "ADBE Exposure2") {
                var ex = fx.property("ADBE Exposure2-0003"); if (ex.numKeys > 0) { out.brightness = ex.numKeys; }
            } else {
                for (var j = 1; j <= fx.numProperties; j++) {
                    var pp = fx.property(j);
                    if (pp && pp.numKeys > 0) { out["effects." + (i - 1) + "." + pp.name] = pp.numKeys; }
                }
            }
        }
    } catch (e3) { }
    return out;
};

AE.layerSummary = function (layer, index, depth) {
    depth = depth || 0;
    var s = { index: index, name: String(layer.name), enabled: !!layer.enabled,
              in_s: layer.inPoint, out_s: layer.outPoint };
    if (layer instanceof TextLayer) {
        s.kind = "text";
        var td = layer.property("Source Text").value;
        s.text = String(td.text);
        s.font = String(td.font);
        s.font_size = td.fontSize;
        try { s.stretch = [td.horizontalScale, td.verticalScale]; } catch (e0) { }
        try { s.stroke = td.applyStroke ? { width: td.strokeWidth } : null; } catch (e0b) { }
        try { s.fill = !!td.applyFill; } catch (e0c) { }
    } else if (layer instanceof ShapeLayer) {
        s.kind = "shape";
    } else if (layer instanceof AVLayer && layer.source && layer.source instanceof CompItem) {
        s.kind = "precomp";
        s.source = String(layer.source.name);
        if (depth < 1 && layer.source.numLayers > 0) { s.inner = AE.layerSummary(layer.source.layer(1), 1, depth + 1); }
    } else if (layer instanceof AVLayer && layer.source && layer.source instanceof FootageItem) {
        s.kind = "footage";
        s.source = layer.source.file ? String(layer.source.file.fsName) : String(layer.source.name);
        s.footage_missing = !!(layer.source.footageMissing);
    } else {
        s.kind = "other";
    }
    try { s.motion_blur = !!layer.motionBlur; } catch (e1) { }
    try { s.blend = layer.blendingMode; } catch (e2) { }
    try { s.masks = layer.property("ADBE Mask Parade").numProperties; } catch (e3) { }
    try {
        var fxn = [];
        var parade = layer.property("ADBE Effect Parade");
        for (var i = 1; i <= parade.numProperties; i++) { fxn.push(String(parade.property(i).matchName)); }
        s.effects = fxn;
    } catch (e4) { }
    try {
        if (layer.trackMatteLayer) { s.matte = { layer: String(layer.trackMatteLayer.name), type: layer.trackMatteType }; }
    } catch (e5) { }
    s.keys = AE.keyCounts(layer);
    var pos = layer.property("Position");
    if (pos) { s.position_keys = pos.numKeys; }
    var op = layer.property("Opacity");
    if (op) { s.opacity_keys = op.numKeys; }
    return s;
};

AE.compSummary = function (comp) {
    var layers = [];
    for (var i = 1; i <= comp.numLayers; i++) { layers.push(AE.layerSummary(comp.layer(i), i)); }
    var out = { name: String(comp.name), width: comp.width, height: comp.height,
                fps: comp.frameRate, duration_s: comp.duration,
                frames: Math.round(comp.duration * comp.frameRate), layers: layers };
    try { out.motion_blur = !!comp.motionBlur; out.shutter_angle = comp.shutterAngle; out.shutter_phase = comp.shutterPhase; } catch (e) { }
    return out;
};

