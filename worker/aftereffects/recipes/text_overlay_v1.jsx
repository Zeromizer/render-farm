/*
 * Recipe text_overlay_v1: a transparent overlay composition with editable
 * text layers, shape layers and optional imported images, animated with
 * position / opacity / scale keyframes. Built-in features only (no plugins).
 *
 * Input: $.global.AE_JOB, written by worker/aftereffects/pipeline.py from a
 * request that already passed schema.validate_request(). The shape is:
 *
 *   { manifest_path, project_path, recipe, recipe_revision,
 *     settings: { composition: {name,width,height,fps,duration_s},
 *                 texts: [...], shapes: [...], images: [...] },
 *     assets: { <name>: "<absolute path inside the job workspace>" },
 *     om_template: "Lossless with Alpha" }
 *
 * Output: the saved .aep at project_path and a JSON manifest at manifest_path
 * ({ok:true,...} or {ok:false,error:{code,message,line,file}}).
 *
 * Nothing here reads outside AE_JOB.assets, and nothing here is evaluated
 * from the request: text values are data, never code.
 */

(function () {
    var job = $.global.AE_JOB;
    var manifest = { ok: false, stage: "author", recipe: job.recipe, recipe_revision: job.recipe_revision,
                     started_at: (new Date()).toString(), warnings: [] };
    var t0 = new Date().getTime();

    function finish(extra) {
        for (var k in extra) { if (extra.hasOwnProperty(k)) { manifest[k] = extra[k]; } }
        manifest.elapsed_ms = new Date().getTime() - t0;
        AE.writeManifest(job.manifest_path, manifest);
    }

    function setTiming(layer, spec, compDuration) {
        var inS = (spec.in_s !== undefined) ? spec.in_s : 0;
        var outS = (spec.out_s !== undefined) ? spec.out_s : compDuration;
        if (outS > compDuration) { outS = compDuration; }
        // Measured on AE 26.5: setting inPoint on a layer without a fixed
        // source duration SHIFTS it (outPoint moves with it), so the out point
        // is trimmed after the in point, never before.
        layer.startTime = 0;
        layer.inPoint = inS;
        layer.outPoint = outS;
        if (Math.abs(layer.inPoint - inS) > 0.001 || Math.abs(layer.outPoint - outS) > 0.001) {
            AE.fail("VERIFY_FAILED", "layer '" + layer.name + "' timing landed at " + layer.inPoint + ".." + layer.outPoint +
                    " instead of " + inS + ".." + outS);
        }
        return { in_s: inS, out_s: outS };
    }

    function ease(prop, keyIndex) {
        try {
            prop.setInterpolationTypeAtKey(keyIndex, KeyframeInterpolationType.BEZIER, KeyframeInterpolationType.BEZIER);
            prop.setTemporalEaseAtKey(keyIndex, [new KeyframeEase(0, 33.3)], [new KeyframeEase(0, 33.3)]);
        } catch (e) { manifest.warnings.push("ease failed on " + prop.name + ": " + e.message); }
    }

    function animate(layer, spec, timing) {
        var fadeIn = spec.fade_in_s || 0;
        var fadeOut = spec.fade_out_s || 0;
        var op = layer.property("Opacity");
        var target = (spec.opacity !== undefined) ? spec.opacity : 100;
        if (fadeIn > 0 || fadeOut > 0) {
            if (fadeIn > 0) {
                op.setValueAtTime(timing.in_s, 0);
                op.setValueAtTime(timing.in_s + fadeIn, target);
            } else {
                op.setValueAtTime(timing.in_s, target);
            }
            if (fadeOut > 0) {
                op.setValueAtTime(timing.out_s - fadeOut, target);
                op.setValueAtTime(timing.out_s, 0);
            }
            for (var k = 1; k <= op.numKeys; k++) { ease(op, k); }
        } else {
            op.setValue(target);
        }

        var pos = layer.property("Position");
        var base = spec.position;
        if (spec.slide_from && (spec.slide_from[0] !== 0 || spec.slide_from[1] !== 0)) {
            var dur = spec.slide_s || fadeIn || 0.4;
            pos.setValueAtTime(timing.in_s, [base[0] + spec.slide_from[0], base[1] + spec.slide_from[1]]);
            pos.setValueAtTime(timing.in_s + dur, [base[0], base[1]]);
            for (var pk = 1; pk <= pos.numKeys; pk++) { ease(pos, pk); }
        } else {
            pos.setValue([base[0], base[1]]);
        }

        var scale = layer.property("Scale");
        var s = (spec.scale !== undefined) ? spec.scale : 100;
        if (spec.scale_from !== undefined && spec.scale_from !== s) {
            var sdur = spec.scale_s || fadeIn || 0.4;
            scale.setValueAtTime(timing.in_s, [spec.scale_from, spec.scale_from]);
            scale.setValueAtTime(timing.in_s + sdur, [s, s]);
            for (var sk = 1; sk <= scale.numKeys; sk++) { ease(scale, sk); }
        } else if (s !== 100) {
            scale.setValue([s, s]);
        }
    }

    function addShadow(layer, spec) {
        if (!spec.shadow) { return; }
        var effects = layer.property("ADBE Effect Parade");
        if (!effects.canAddProperty("ADBE Drop Shadow")) {
            AE.fail("EFFECT_MISSING", "built-in Drop Shadow (ADBE Drop Shadow) is not available");
        }
        var fx = effects.addProperty("ADBE Drop Shadow");
        fx.property("ADBE Drop Shadow-0002").setValue(spec.shadow.opacity !== undefined ? spec.shadow.opacity * 2.55 : 128);
        fx.property("ADBE Drop Shadow-0004").setValue(spec.shadow.distance !== undefined ? spec.shadow.distance : 8);
        fx.property("ADBE Drop Shadow-0005").setValue(spec.shadow.softness !== undefined ? spec.shadow.softness : 20);
    }

    try {
        app.beginSuppressDialogs();
        AE.enableFileAccess();
        AE.assertIsolated(job.token);
        manifest.ae = AE.versionInfo();

        var cs = job.settings.composition;
        var comp = app.project.items.addComp(job.composition, cs.width, cs.height, 1.0, cs.duration_s, cs.fps);
        comp.bgColor = [0, 0, 0];
        var textProps = [];
        var fontChecks = {};

        // Texts: bottom-up in the spec means the first entry ends up lowest.
        for (var i = 0; i < job.settings.texts.length; i++) {
            var ts = job.settings.texts[i];
            var tl = comp.layers.addText(String(ts.text).replace(/\n/g, "\r"));
            tl.name = ts.id;
            var prop = tl.property("Source Text");
            var fc = AE.fontAvailable(ts.font, prop);
            fontChecks[ts.font] = fc;
            if (!fc.available) {
                AE.fail("FONT_MISSING", "font '" + ts.font + "' is not installed on this After Effects host" +
                        (fc.resolved ? " (resolved to '" + fc.resolved + "')" : ""));
            }
            var td = prop.value;
            td.font = ts.font;
            td.fontSize = ts.size;
            td.fillColor = AE.hexToRgb(ts.color);
            td.applyFill = true;
            td.applyStroke = false;
            if (ts.tracking !== undefined) { td.tracking = ts.tracking; }
            // ExtendScript mis-parses nested ternaries (a ? b : c ? d : e), which
            // silently turned "left" into right-justified text before 2026-09-11.
            td.justification = AE.justification(ts.justify || "center");
            prop.setValue(td);
            var timing = setTiming(tl, ts, cs.duration_s);
            animate(tl, ts, timing);
            addShadow(tl, ts);
            textProps.push({ id: ts.id, index: tl.index });
        }

        // Shapes: rectangles / ellipses with a solid fill, centred on position.
        for (var j = 0; j < job.settings.shapes.length; j++) {
            var ss = job.settings.shapes[j];
            var sl = comp.layers.addShape();
            sl.name = ss.id;
            var group = sl.property("ADBE Root Vectors Group").addProperty("ADBE Vector Group");
            group.name = ss.id + "_group";
            var vectors = group.property("ADBE Vectors Group");
            var shape;
            if (ss.kind === "ellipse") {
                shape = vectors.addProperty("ADBE Vector Shape - Ellipse");
                shape.property("ADBE Vector Ellipse Size").setValue([ss.size[0], ss.size[1]]);
            } else {
                shape = vectors.addProperty("ADBE Vector Shape - Rect");
                shape.property("ADBE Vector Rect Size").setValue([ss.size[0], ss.size[1]]);
                shape.property("ADBE Vector Rect Roundness").setValue(ss.radius || 0);
            }
            var fill = vectors.addProperty("ADBE Vector Graphic - Fill");
            fill.property("ADBE Vector Fill Color").setValue(AE.hexToRgb(ss.color).concat([1]));
            var stiming = setTiming(sl, ss, cs.duration_s);
            animate(sl, ss, stiming);
            addShadow(sl, ss);
        }

        // Images: footage from the job's own staged assets.
        for (var m = 0; m < job.settings.images.length; m++) {
            var is = job.settings.images[m];
            var path = job.assets[is.asset];
            var file = new File(path);
            if (!file.exists) { AE.fail("ASSET_MISSING", "asset '" + is.asset + "' not found at " + path); }
            var io = new ImportOptions(file);
            if (!io.canImportAs(ImportAsType.FOOTAGE)) { AE.fail("ASSET_MISSING", "asset '" + is.asset + "' cannot be imported as footage"); }
            io.importAs = ImportAsType.FOOTAGE;
            var item = app.project.importFile(io);
            item.name = is.asset;
            var il = comp.layers.add(item);
            il.name = is.id;
            var itiming = setTiming(il, is, cs.duration_s);
            animate(il, is, itiming);
            addShadow(il, is);
        }

        // Layer order: the spec's z order is texts over shapes over images.
        // AE stacks the most recently added layer on top, so images were added
        // last and sit on top; move every image to the bottom, then shapes.
        for (var z = comp.numLayers; z >= 1; z--) {
            var L = comp.layer(z);
            if (L instanceof AVLayer && !(L instanceof TextLayer) && !(L instanceof ShapeLayer)) { L.moveToEnd(); }
        }
        for (var z2 = comp.numLayers; z2 >= 1; z2--) {
            var L2 = comp.layer(z2);
            if (L2 instanceof ShapeLayer) {
                // move below every text layer: find the first text from the bottom
                var lowestText = 0;
                for (var q = comp.numLayers; q >= 1; q--) { if (comp.layer(q) instanceof TextLayer) { lowestText = q; break; } }
                if (lowestText && L2.index < lowestText) { L2.moveAfter(comp.layer(lowestText)); }
            }
        }

        var templates = AE.outputModuleTemplates(comp);
        var templateOk = AE.contains(templates, job.om_template);
        if (!templateOk) {
            AE.fail("OM_TEMPLATE_MISSING", "output module template '" + job.om_template + "' is not defined in this After Effects; available: " + templates.join(" | "));
        }

        var out = new File(job.project_path);
        app.project.save(out);
        if (!out.exists) { AE.fail("JSX_ERROR", "project save reported success but " + job.project_path + " is missing"); }

        finish({ ok: true, project_file: String(out.fsName), comp: AE.compSummary(comp),
                 om_templates: templates, om_template: job.om_template, fonts: fontChecks });
    } catch (e) {
        try {
            finish({ ok: false, error: AE.errorInfo(e, e.aeCode) });
        } catch (e2) { /* nothing left to report through */ }
    } finally {
        try { app.endSuppressDialogs(false); } catch (e3) { }
        try { app.project.dirty = false; } catch (e4) { }
        AE.quitIfIsolated();
    }
})();
