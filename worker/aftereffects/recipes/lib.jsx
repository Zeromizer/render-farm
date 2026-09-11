/*
 * Shared ExtendScript helpers for the Aimotion After Effects recipes.
 *
 * ExtendScript is ES3: no JSON, no Array.indexOf, no trailing commas, no let.
 * Everything here is written for that dialect. The Python driver evaluates
 * this file with $.evalFile() before the recipe, after it has put the
 * validated job object at $.global.AE_JOB.
 */

var AE = AE || {};

AE.jsonStr = function (v, indent) {
    indent = indent || "";
    var t = typeof v;
    if (v === null || v === undefined) { return "null"; }
    if (t === "number") { return isFinite(v) ? String(v) : "null"; }
    if (t === "boolean") { return v ? "true" : "false"; }
    if (t === "string") { return AE.jsonEscape(v); }
    if (v instanceof Array) {
        var parts = [];
        for (var i = 0; i < v.length; i++) { parts.push(AE.jsonStr(v[i], indent + "  ")); }
        return "[" + parts.join(",") + "]";
    }
    if (t === "object") {
        var keys = [];
        for (var k in v) { if (v.hasOwnProperty(k) && typeof v[k] !== "function") { keys.push(k); } }
        var out = [];
        for (var j = 0; j < keys.length; j++) {
            out.push(AE.jsonEscape(keys[j]) + ":" + AE.jsonStr(v[keys[j]], indent + "  "));
        }
        return "{" + out.join(",") + "}";
    }
    return AE.jsonEscape(String(v));
};

AE.jsonEscape = function (s) {
    s = String(s);
    var out = "\"";
    for (var i = 0; i < s.length; i++) {
        var c = s.charAt(i);
        var code = s.charCodeAt(i);
        if (c === "\"") { out += "\\\""; }
        else if (c === "\\") { out += "\\\\"; }
        else if (c === "\n") { out += "\\n"; }
        else if (c === "\r") { out += "\\r"; }
        else if (c === "\t") { out += "\\t"; }
        else if (code < 32) {
            var hex = code.toString(16);
            while (hex.length < 4) { hex = "0" + hex; }
            out += "\\u" + hex;
        } else { out += c; }
    }
    return out + "\"";
};

/* Write a UTF-8 text file; returns true on success. Needs the
 * "Allow Scripts to Write Files and Access Network" preference. */
AE.writeText = function (path, text) {
    var f = new File(path);
    f.encoding = "UTF-8";
    f.lineFeed = "Unix";
    if (!f.open("w")) { return false; }
    var ok = f.write(text);
    f.close();
    return ok;
};

AE.writeManifest = function (path, obj) {
    obj.written_at = (new Date()).toString();
    if (!AE.writeText(path, AE.jsonStr(obj))) {
        throw new Error("could not write manifest to " + path + " (enable 'Allow Scripts to Write Files and Access Network')");
    }
};

/* Build the error object the Python side reports back (line/message/file). */
AE.errorInfo = function (e, code) {
    var info = { code: code || "JSX_ERROR", message: String(e && e.message ? e.message : e) };
    if (e && e.line !== undefined) { info.line = e.line; }
    if (e && e.fileName !== undefined) { info.file = String(e.fileName); }
    if (e && e.name !== undefined) { info.name = String(e.name); }
    return info;
};

/* Raise an error with a machine-readable code the manifest will carry. */
AE.fail = function (code, message) {
    var e = new Error(message);
    e.aeCode = code;
    throw e;
};

AE.hexToRgb = function (hex) {
    hex = String(hex).replace("#", "");
    if (hex.length === 3) { hex = hex.charAt(0) + hex.charAt(0) + hex.charAt(1) + hex.charAt(1) + hex.charAt(2) + hex.charAt(2); }
    return [parseInt(hex.substr(0, 2), 16) / 255, parseInt(hex.substr(2, 2), 16) / 255, parseInt(hex.substr(4, 2), 16) / 255];
};

AE.contains = function (arr, v) {
    for (var i = 0; i < arr.length; i++) { if (arr[i] === v) { return true; } }
    return false;
};

/* Enable script file access for this session, without touching anything
 * else in the preferences. Documented in docs/aftereffects-platform-contract.md. */
AE.enableFileAccess = function () {
    // Best effort, in-memory only. savePrefAsLong on "Main Pref Section v2"
    // is refused ("Scripts may not change security preferences"); the legacy
    // section accepts it. NEVER call app.preferences.saveToDisk()/reload()
    // here: measured on AE 26.5, reload() drops the in-memory permission and
    // every File.write afterwards silently produces a 0-byte file.
    var sections = ["Main Pref Section v2", "Main Pref Section"];
    for (var i = 0; i < sections.length; i++) {
        try {
            app.preferences.savePrefAsLong(sections[i], "Pref_SCRIPTING_FILE_NETWORK_SECURITY", 1);
            return;
        } catch (e) { }
    }
};

/* The project isolation guard. The driver launches AfterFX.exe -m -noui -r
 * with a one-off AE_JOB_TOKEN in that process's environment. If this script
 * was handed to an instance that was already running (an operator's), the
 * environment lacks the token and we stop before touching the project.
 * AE.isolated stays false then, and the recipes only app.quit() when it is
 * true, so a stray run never closes somebody's session. */
AE.isolated = false;
AE.assertIsolated = function (token) {
    var seen = null;
    try { seen = $.getenv("AE_JOB_TOKEN"); } catch (e) { }
    if (!token || !seen || String(seen) !== String(token)) {
        AE.fail("AE_NOT_ISOLATED", "this After Effects instance was not launched for this job (AE_JOB_TOKEN mismatch); refusing to touch its project");
    }
    var p = app.project;
    if (p.file !== null) { AE.fail("AE_NOT_ISOLATED", "a saved project is open in this instance: " + p.file.fsName); }
    if (p.numItems > 0) { AE.fail("AE_NOT_ISOLATED", "the instance already holds " + p.numItems + " project items"); }
    AE.isolated = true;
};

AE.quitIfIsolated = function () {
    if (AE.isolated) { app.quit(); }
};

AE.versionInfo = function () {
    var info = { app_version: String(app.version) };
    try { info.build_name = String(app.buildName); } catch (e1) { }
    try { info.build_number = String(app.buildNumber); } catch (e2) { }
    try { info.language = String(app.isoLanguage); } catch (e3) { }
    return info;
};

/* PostScript-name font check. AE 24+ exposes app.fonts; older builds fall
 * back to setting the font on a scratch TextDocument and reading it back. */
AE.fontAvailable = function (psName, textProp) {
    try {
        if (app.fonts && app.fonts.getFontsByPostScriptName) {
            var hits = app.fonts.getFontsByPostScriptName(psName);
            return { available: hits.length > 0, method: "app.fonts" };
        }
    } catch (e) { }
    var doc = textProp.value;
    doc.font = psName;
    textProp.setValue(doc);
    var back = textProp.value.font;
    return { available: String(back) === String(psName), method: "readback", resolved: String(back) };
};

/* The names of the output-module templates this install offers, measured on a
 * scratch render-queue item that is removed again. */
AE.outputModuleTemplates = function (comp) {
    var rq = app.project.renderQueue;
    var item = rq.items.add(comp);
    var names = [];
    try {
        var om = item.outputModule(1);
        var t = om.templates;
        for (var i = 0; i < t.length; i++) { names.push(String(t[i])); }
    } finally {
        item.remove();
    }
    return names;
};

AE.layerSummary = function (layer, index) {
    var s = { index: index, name: String(layer.name), enabled: !!layer.enabled,
              in_s: layer.inPoint, out_s: layer.outPoint };
    if (layer instanceof TextLayer) {
        s.kind = "text";
        var td = layer.property("Source Text").value;
        s.text = String(td.text);
        s.font = String(td.font);
        s.font_size = td.fontSize;
    } else if (layer instanceof ShapeLayer) {
        s.kind = "shape";
    } else if (layer instanceof AVLayer && layer.source && layer.source instanceof FootageItem) {
        s.kind = "footage";
        s.source = layer.source.file ? String(layer.source.file.fsName) : String(layer.source.name);
        s.footage_missing = !!(layer.source.footageMissing);
    } else {
        s.kind = "other";
    }
    var pos = layer.property("Position");
    if (pos) { s.position_keys = pos.numKeys; }
    var op = layer.property("Opacity");
    if (op) { s.opacity_keys = op.numKeys; }
    return s;
};

AE.compSummary = function (comp) {
    var layers = [];
    for (var i = 1; i <= comp.numLayers; i++) { layers.push(AE.layerSummary(comp.layer(i), i)); }
    return { name: String(comp.name), width: comp.width, height: comp.height,
             fps: comp.frameRate, duration_s: comp.duration,
             frames: Math.round(comp.duration * comp.frameRate), layers: layers };
};

AE.findComp = function (name) {
    for (var i = 1; i <= app.project.numItems; i++) {
        var it = app.project.item(i);
        if (it instanceof CompItem && it.name === name) { return it; }
    }
    return null;
};
