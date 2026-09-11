/*
 * Controlled revision of a text_overlay_v1 project: change the text and/or
 * timing of named layers, save under a new path. This is the "edit one value
 * and re-render" path the platform's order versioning maps onto.
 *
 * Input: $.global.AE_JOB = { manifest_path, source_project_path, project_path,
 *                            composition,
 *                            changes: [ { id, text?, in_s?, out_s? }, ... ] }
 */
(function () {
    var job = $.global.AE_JOB;
    var manifest = { ok: false, stage: "revise", started_at: (new Date()).toString(), applied: [] };
    try {
        app.beginSuppressDialogs();
        AE.enableFileAccess();
        AE.assertIsolated(job.token);
        manifest.ae = AE.versionInfo();
        var src = new File(job.source_project_path);
        if (!src.exists) { AE.fail("ASSET_MISSING", "source project not found: " + job.source_project_path); }
        app.open(src);
        var comp = AE.findComp(job.composition);
        if (!comp) { AE.fail("VERIFY_FAILED", "composition '" + job.composition + "' not found"); }

        for (var c = 0; c < job.changes.length; c++) {
            var ch = job.changes[c];
            var layer = null;
            for (var i = 1; i <= comp.numLayers; i++) {
                if (comp.layer(i).name === ch.id) { layer = comp.layer(i); break; }
            }
            if (!layer) { AE.fail("VERIFY_FAILED", "layer '" + ch.id + "' not found in composition"); }
            var applied = { id: ch.id };
            if (ch.text !== undefined) {
                if (!(layer instanceof TextLayer)) { AE.fail("INVALID_REQUEST", "layer '" + ch.id + "' is not a text layer"); }
                var prop = layer.property("Source Text");
                var td = prop.value;
                td.text = String(ch.text).replace(/\n/g, "\r");
                prop.setValue(td);
                applied.text = String(prop.value.text);
            }
            if (ch.in_s !== undefined || ch.out_s !== undefined) {
                var inS = (ch.in_s !== undefined) ? ch.in_s : layer.inPoint;
                var outS = (ch.out_s !== undefined) ? ch.out_s : layer.outPoint;
                if (outS > comp.duration) { outS = comp.duration; }
                // Shift keyframes with the in point so fades stay attached to the edges.
                var delta = inS - layer.inPoint;
                var names = ["Opacity", "Position", "Scale"];
                for (var n = 0; n < names.length; n++) {
                    var p = layer.property(names[n]);
                    if (!p || p.numKeys === 0) { continue; }
                    var keys = [];
                    for (var k = 1; k <= p.numKeys; k++) { keys.push({ t: p.keyTime(k), v: p.keyValue(k) }); }
                    while (p.numKeys > 0) { p.removeKey(1); }
                    var oldOut = layer.outPoint;
                    for (var k2 = 0; k2 < keys.length; k2++) {
                        var t = keys[k2].t + delta;
                        // keys that hung on the old out point follow the new one
                        if (Math.abs(keys[k2].t - oldOut) < 1e-6) { t = outS; }
                        else if (keys[k2].t > oldOut - 2 && keys[k2].t < oldOut) { t = outS - (oldOut - keys[k2].t); }
                        p.setValueAtTime(t, keys[k2].v);
                    }
                }
                layer.inPoint = inS;
                layer.outPoint = outS;
                if (Math.abs(layer.inPoint - inS) > 0.001 || Math.abs(layer.outPoint - outS) > 0.001) {
                    AE.fail("VERIFY_FAILED", "layer '" + ch.id + "' timing landed at " + layer.inPoint + ".." + layer.outPoint +
                            " instead of " + inS + ".." + outS);
                }
                applied.in_s = layer.inPoint;
                applied.out_s = layer.outPoint;
            }
            manifest.applied.push(applied);
        }

        var out = new File(job.project_path);
        app.project.save(out);
        if (!out.exists) { AE.fail("JSX_ERROR", "save reported success but " + job.project_path + " is missing"); }
        manifest.ok = true;
        manifest.project_file = String(out.fsName);
        manifest.comp = AE.compSummary(comp);
        AE.writeManifest(job.manifest_path, manifest);
    } catch (e) {
        try { manifest.error = AE.errorInfo(e, e.aeCode); AE.writeManifest(job.manifest_path, manifest); } catch (e2) { }
    } finally {
        try { app.endSuppressDialogs(false); } catch (e3) { }
        try { app.project.dirty = false; } catch (e4) { }
        AE.quitIfIsolated();
    }
})();
