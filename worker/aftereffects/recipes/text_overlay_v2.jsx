/*
 * Recipe text_overlay_v2 (docs/aftereffects-text-overlay-v2.md): ordered
 * layers, frame-aligned timing, anchors, independent scale, text stretch and
 * stroke, comp-space clip (precompose + mask), track mattes, built-in effects,
 * keyframes with easing, comp motion blur. Built-in features only.
 *
 * Input: $.global.AE_JOB as for v1, with settings = {composition, layers}
 * already normalized by schema_v2.py (frames AND seconds present, anchors
 * explicit, ease as "linear" | "hold" | {in:[speed,influence], out:[...]}).
 */

(function () {
    var job = $.global.AE_JOB;
    var manifest = { ok: false, stage: "author", recipe: job.recipe, recipe_revision: job.recipe_revision,
                     started_at: (new Date()).toString(), warnings: [], easing_resolved: {}, anchors_resolved: {} };
    var t0 = new Date().getTime();

    function finish(extra) {
        for (var k in extra) { if (extra.hasOwnProperty(k)) { manifest[k] = extra[k]; } }
        manifest.elapsed_ms = new Date().getTime() - t0;
        AE.writeManifest(job.manifest_path, manifest);
    }

    // The last step reached, so a JSX error names the layer and stage even
    // when AE's line number points elsewhere.
    function step(s) {
        manifest.step = s;
        // Also persisted next to the manifest so a hard AE crash still names the step.
        try { var sf = new File(job.manifest_path + ".step"); sf.open("w"); sf.write(s); sf.close(); } catch (e) { }
    }

    function rgba(hex) { return AE.hexToRgb(hex).concat([1]); }

    function setTiming(layer, L) {
        layer.startTime = 0;
        layer.inPoint = L.in_s;
        layer.outPoint = L.out_s;
        if (Math.abs(layer.inPoint - L.in_s) > 0.001 || Math.abs(layer.outPoint - L.out_s) > 0.001) {
            AE.fail("VERIFY_FAILED", "layer '" + L.id + "' timing landed at " + layer.inPoint + ".." + layer.outPoint + " instead of " + L.in_s + ".." + L.out_s);
        }
    }

    function propertyByName(group, name) {
        var p = group.property(name);
        if (!p) { AE.fail("EFFECT_MISSING", "property '" + name + "' not found on " + group.name); }
        return p;
    }

    // ---------------------------------------------------------------- layers
    function makeText(comp, L) {
        var tl = L.box ? comp.layers.addBoxText([L.box.width, L.box.height], L.text.replace(/\n/g, "\r"))
                       : comp.layers.addText(L.text.replace(/\n/g, "\r"));
        var prop = tl.property("Source Text");
        step(L.id + ": font check");
        var fc = AE.fontAvailable(L.font, prop);
        manifest.fonts = manifest.fonts || {};
        manifest.fonts[L.font] = fc;
        if (!fc.available) { AE.fail("FONT_MISSING", "font '" + L.font + "' is not installed on this After Effects host"); }
        step(L.id + ": build text document");
        var td = prop.value;
        td.font = L.font;
        td.fontSize = L.size;
        td.applyFill = !!L.fill;
        if (L.fill) { td.fillColor = AE.hexToRgb(L.color); }
        if (L.stroke) {
            td.applyStroke = true;
            td.strokeColor = AE.hexToRgb(L.stroke.color);
            td.strokeWidth = L.stroke.width;
            td.strokeOverFill = !!L.stroke.over_fill;
        } else { td.applyStroke = false; }
        if (L.tracking !== undefined) { td.tracking = L.tracking; }
        if (L.leading !== undefined) { td.autoLeading = false; td.leading = L.leading; }
        if (L.baseline_shift !== undefined) { td.baselineShift = L.baseline_shift; }
        // No nested ternaries anywhere in these recipes: ExtendScript parses
        // a ? b : c ? d : e as (a ? b : c) ? d : e (measured on AE 26.5).
        td.justification = AE.justification(L.justify);
        if (L.stretch[0] !== 1 || L.stretch[1] !== 1) {
            td.horizontalScale = L.stretch[0];
            td.verticalScale = L.stretch[1];
        }
        step(L.id + ": setValue text document");
        prop.setValue(td);
        step(L.id + ": readback");
        var back = prop.value;
        step(L.id + ": readback stretch check");
        if (L.stretch[0] !== 1 || L.stretch[1] !== 1) {
            if (Math.abs(back.horizontalScale - L.stretch[0]) > 0.01 || Math.abs(back.verticalScale - L.stretch[1]) > 0.01) {
                AE.fail("VERIFY_FAILED", "layer '" + L.id + "' stretch read back as [" + back.horizontalScale + ", " + back.verticalScale + "]");
            }
        }
        step(L.id + ": readback stroke check");
        if (L.stroke && !back.applyStroke) { AE.fail("VERIFY_FAILED", "layer '" + L.id + "' stroke did not apply"); }
        step(L.id + ": text done");
        return tl;
    }

    function makeShape(comp, L) {
        var sl = comp.layers.addShape();
        var group = sl.property("ADBE Root Vectors Group").addProperty("ADBE Vector Group");
        group.name = L.id + "_group";
        var vectors = group.property("ADBE Vectors Group");
        var shape;
        if (L.shape === "ellipse") {
            shape = vectors.addProperty("ADBE Vector Shape - Ellipse");
            shape.property("ADBE Vector Ellipse Size").setValue([L.size[0], L.size[1]]);
        } else {
            shape = vectors.addProperty("ADBE Vector Shape - Rect");
            shape.property("ADBE Vector Rect Size").setValue([L.size[0], L.size[1]]);
            shape.property("ADBE Vector Rect Roundness").setValue(L.radius || 0);
        }
        if (L.stroke) {
            var st = vectors.addProperty("ADBE Vector Graphic - Stroke");
            st.property("ADBE Vector Stroke Color").setValue(rgba(L.stroke.color));
            st.property("ADBE Vector Stroke Width").setValue(L.stroke.width);
        }
        if (L.color) {
            var fill = vectors.addProperty("ADBE Vector Graphic - Fill");
            fill.property("ADBE Vector Fill Color").setValue(rgba(L.color));
        }
        return sl;
    }

    function makeImage(comp, L) {
        var path = job.assets[L.asset];
        var file = new File(path);
        if (!file.exists) { AE.fail("ASSET_MISSING", "asset '" + L.asset + "' not found at " + path); }
        var io = new ImportOptions(file);
        if (!io.canImportAs(ImportAsType.FOOTAGE)) { AE.fail("ASSET_MISSING", "asset '" + L.asset + "' cannot be imported as footage"); }
        io.importAs = ImportAsType.FOOTAGE;
        var item = app.project.importFile(io);
        item.name = L.asset;
        var il = comp.layers.add(item);
        if (L.size) {
            var sx = 100 * L.size[0] / item.width, sy = 100 * L.size[1] / item.height;
            if (L.fit !== "stretch") { sx = sy = Math.min(sx, sy); }
            il.property("Scale").setValue([sx, sy]);
            L._fitScale = [sx, sy];
        }
        return il;
    }

    // ---------------------------------------------------------------- anchor + transform
    function resolveAnchor(layer, L) {
        var r = layer.sourceRectAtTime(L.in_s, false);
        if (!(r.width > 0 && r.height > 0)) { AE.fail("ANCHOR_UNRESOLVED", "layer '" + L.id + "' has empty bounds at frame " + L.in_f); }
        var ax, ay;
        if (typeof L.anchor.x === "number") { ax = L.anchor.x; }
        else if (L.anchor.x === "left") { ax = r.left; }
        else if (L.anchor.x === "right") { ax = r.left + r.width; }
        else { ax = r.left + r.width / 2; }
        if (typeof L.anchor.y === "number") { ay = L.anchor.y; }
        else if (L.anchor.y === "baseline") { ay = 0; }
        else if (L.anchor.y === "top") { ay = r.top; }
        else if (L.anchor.y === "bottom") { ay = r.top + r.height; }
        else { ay = r.top + r.height / 2; }
        layer.property("Anchor Point").setValue([ax, ay]);
        manifest.anchors_resolved[L.id] = { anchor: [ax, ay], bounds: { left: r.left, top: r.top, width: r.width, height: r.height } };
    }

    function applyTransform(layer, L) {
        layer.property("Position").setValue([L.position[0], L.position[1]]);
        var sc = layer.property("Scale");
        if (L._fitScale) { sc.setValue([L._fitScale[0] * L.scale[0] / 100, L._fitScale[1] * L.scale[1] / 100]); }
        else if (L.scale[0] !== 100 || L.scale[1] !== 100) { sc.setValue([L.scale[0], L.scale[1]]); }
        if (L.rotation) { layer.property("Rotation").setValue(L.rotation); }
        layer.property("Opacity").setValue(L.opacity);
        if (L.blend !== "normal") { layer.blendingMode = BlendingMode[({ multiply: "MULTIPLY", screen: "SCREEN", add: "ADD", overlay: "OVERLAY", lighten: "LIGHTEN", darken: "DARKEN" })[L.blend]]; }
        layer.motionBlur = !!L.motion_blur;
    }

    // ---------------------------------------------------------------- effects
    var EFFECT_PARAMS = {
        glow: function (fx, e) {
            fx.property("ADBE Glo2-0001").setValue(e.based_on === "color" ? 1 : 2);   // single ternary is fine
            fx.property("ADBE Glo2-0002").setValue(e.threshold);
            fx.property("ADBE Glo2-0003").setValue(e.radius);
            fx.property("ADBE Glo2-0004").setValue(e.intensity);
            fx.property("ADBE Glo2-0005").setValue(e.composite === "behind" ? 2 : 1);
            fx.property("ADBE Glo2-0007").setValue(2);
            fx.property("ADBE Glo2-0012").setValue(rgba(e.color));
            fx.property("ADBE Glo2-0013").setValue(rgba(e.color));
        },
        bevel_highlight: function (fx, e) {
            fx.property("ADBE Bevel Alpha-0001").setValue(e.thickness);
            fx.property("ADBE Bevel Alpha-0002").setValue(e.angle);
            fx.property("ADBE Bevel Alpha-0003").setValue(rgba(e.color));
            fx.property("ADBE Bevel Alpha-0004").setValue(e.intensity);
        },
        drop_shadow: function (fx, e) {
            fx.property("ADBE Drop Shadow-0001").setValue(rgba(e.color));
            fx.property("ADBE Drop Shadow-0002").setValue(e.opacity * 2.55);
            fx.property("ADBE Drop Shadow-0003").setValue(e.direction);
            fx.property("ADBE Drop Shadow-0004").setValue(e.distance);
            fx.property("ADBE Drop Shadow-0005").setValue(e.softness);
        },
        gaussian_blur: function (fx, e) { fx.property("ADBE Gaussian Blur 2-0001").setValue(e.radius); },
        fill: function (fx, e) { propertyByName(fx, "Color").setValue(rgba(e.color)); propertyByName(fx, "Opacity").setValue(e.opacity / 100); },
        tint: function (fx, e) { propertyByName(fx, "Map Black To").setValue(rgba(e.black)); propertyByName(fx, "Map White To").setValue(rgba(e.white)); propertyByName(fx, "Amount to Tint").setValue(e.amount); },
        choker: function (fx, e) { propertyByName(fx, "Choke Matte").setValue(e.amount); },
        exposure: function (fx, e) { fx.property("ADBE Exposure2-0003").setValue(e.stops); },
        roughen_edges: function (fx, e) { propertyByName(fx, "Border").setValue(e.border); propertyByName(fx, "Scale").setValue(e.scale); propertyByName(fx, "Complexity").setValue(e.complexity); propertyByName(fx, "Random Seed").setValue(e.seed); }
    };
    // keyframable effect params -> property accessor
    var EFFECT_KEY_PROPS = {
        glow: { radius: "ADBE Glo2-0003", intensity: "ADBE Glo2-0004", threshold: "ADBE Glo2-0002" },
        bevel_highlight: { thickness: "ADBE Bevel Alpha-0001", angle: "ADBE Bevel Alpha-0002", intensity: "ADBE Bevel Alpha-0004" },
        drop_shadow: { opacity: "ADBE Drop Shadow-0002", distance: "ADBE Drop Shadow-0004", softness: "ADBE Drop Shadow-0005", direction: "ADBE Drop Shadow-0003" },
        gaussian_blur: { radius: "ADBE Gaussian Blur 2-0001" },
        fill: { opacity: "Opacity" }, tint: { amount: "Amount to Tint" }, choker: { amount: "Choke Matte" },
        exposure: { stops: "ADBE Exposure2-0003" }, roughen_edges: { border: "Border", scale: "Scale" }
    };
    var EFFECT_KEY_SCALE = { drop_shadow: { opacity: 2.55 } };

    function addEffects(layer, L) {
        var parade = layer.property("ADBE Effect Parade");
        var added = [];
        for (var i = 0; i < L.effects.length; i++) {
            var e = L.effects[i];
            if (!parade.canAddProperty(e.match_name)) { AE.fail("EFFECT_MISSING", "effect " + e.match_name + " (" + e.type + ") is not available"); }
            var fx = parade.addProperty(e.match_name);
            fx.name = e.type + "_" + i;
            EFFECT_PARAMS[e.type](fx, e);
            added.push(fx);
        }
        return added;
    }

    // ---------------------------------------------------------------- keyframes
    function easeArray(prop, spec) {
        var dims = 1;
        var vt = prop.propertyValueType;
        if (vt === PropertyValueType.TwoD) { dims = 2; }
        else if (vt === PropertyValueType.ThreeD) { dims = 3; }
        else if (vt === PropertyValueType.COLOR) { dims = 4; }
        var arr = [];
        for (var d = 0; d < dims; d++) { arr.push(new KeyframeEase(spec[0], spec[1])); }
        return arr;
    }

    function applyKeys(layer, prop, keys, label, transform) {
        for (var i = 0; i < keys.length; i++) {
            var v = keys[i].v;
            if (transform) { v = transform(v); }
            prop.setValueAtTime(keys[i].t, v);
        }
        var resolved = [];
        for (var k = 1; k <= prop.numKeys; k++) {
            var ease = keys[k - 1].ease;
            if (ease === "hold") {
                prop.setInterpolationTypeAtKey(k, KeyframeInterpolationType.LINEAR, KeyframeInterpolationType.HOLD);
            } else if (ease === "linear") {
                prop.setInterpolationTypeAtKey(k, KeyframeInterpolationType.LINEAR, KeyframeInterpolationType.LINEAR);
            } else {
                prop.setInterpolationTypeAtKey(k, KeyframeInterpolationType.BEZIER, KeyframeInterpolationType.BEZIER);
                prop.setTemporalEaseAtKey(k, easeArray(prop, ease["in"]), easeArray(prop, ease.out));
            }
            var r = { f: keys[k - 1].f, ease: ease };
            try {
                var ie = prop.keyInTemporalEase(k)[0], oe = prop.keyOutTemporalEase(k)[0];
                r.ae_in = [ie.speed, ie.influence]; r.ae_out = [oe.speed, oe.influence];
                r.ae_interp = [prop.keyInInterpolationType(k), prop.keyOutInterpolationType(k)];
            } catch (e1) { }
            resolved.push(r);
        }
        manifest.easing_resolved[label] = resolved;
    }

    function addKeyframes(layer, L, effectsAdded) {
        for (var prop in L.keyframes) {
            if (!L.keyframes.hasOwnProperty(prop)) { continue; }
            var keys = L.keyframes[prop];
            var label = L.id + "." + prop;
            if (prop === "position") { applyKeys(layer, layer.property("Position"), keys, label, function (v) { return [v[0], v[1]]; }); }
            else if (prop === "scale") {
                var fs = L._fitScale;
                applyKeys(layer, layer.property("Scale"), keys, label, function (v) { return fs ? [fs[0] * v[0] / 100, fs[1] * v[1] / 100] : [v[0], v[1]]; });
            }
            else if (prop === "rotation") { applyKeys(layer, layer.property("Rotation"), keys, label); }
            else if (prop === "opacity") { applyKeys(layer, layer.property("Opacity"), keys, label); }
            else if (prop === "brightness") {
                var parade = layer.property("ADBE Effect Parade");
                var ex = parade.addProperty("ADBE Exposure2");
                ex.name = "brightness";
                applyKeys(layer, ex.property("ADBE Exposure2-0003"), keys, label, function (v) { return Math.log(v) / Math.LN2; });
            }
            else if (prop === "stretch") {
                var st = layer.property("Source Text");
                for (var i = 0; i < keys.length; i++) {
                    var td = st.valueAtTime(keys[i].t, false);
                    td.horizontalScale = keys[i].v[0]; td.verticalScale = keys[i].v[1];
                    st.setValueAtTime(keys[i].t, td);
                }
                manifest.easing_resolved[label] = "hold (Source Text)";
            }
            else if (prop.indexOf("effects.") === 0) {
                var parts = prop.split(".");
                var idx = parseInt(parts[1], 10), param = parts[2];
                var e = L.effects[idx];
                var acc = EFFECT_KEY_PROPS[e.type][param];
                var target = acc.indexOf("ADBE") === 0 ? effectsAdded[idx].property(acc) : propertyByName(effectsAdded[idx], acc);
                var scale = (EFFECT_KEY_SCALE[e.type] || {})[param] || 1;
                applyKeys(layer, target, keys, label, function (v) { return v * scale; });
            }
        }
    }

    // ---------------------------------------------------------------- clip + matte
    function applyClip(comp, L) {
        var layer = AE.findLayer(comp, L.id);
        var pre = comp.layers.precompose([layer.index], L.id, true);
        var outer = AE.findLayer(comp, L.id);
        if (!outer || !(outer.source instanceof CompItem)) { AE.fail("VERIFY_FAILED", "precompose of '" + L.id + "' did not yield a comp layer"); }
        var mask = outer.property("ADBE Mask Parade").addProperty("ADBE Mask Atom");
        mask.name = "clip";
        var sh = new Shape();
        var c = L.clip;
        sh.vertices = [[c.left, c.top], [c.right, c.top], [c.right, c.bottom], [c.left, c.bottom]];
        sh.closed = true;
        mask.property("ADBE Mask Shape").setValue(sh);
        mask.maskMode = MaskMode.ADD;
        // The precomp layer spans the whole comp by default; trim it to the
        // layer's own window so the project reads the same as the request.
        outer.inPoint = L.in_s;
        outer.outPoint = L.out_s;
        // the inner layer keeps the id so the inspector can find it
        pre.layer(1).name = L.id;
    }

    function applyMatte(comp, L) {
        var layer = AE.findLayer(comp, L.id);
        var matte = AE.findLayer(comp, L.matte.layer);
        if (!matte) { AE.fail("MATTE_TARGET_MISSING", "matte layer '" + L.matte.layer + "' not found for '" + L.id + "'"); }
        var mode = TrackMatteType[({ alpha: "ALPHA", alpha_inverted: "ALPHA_INVERTED", luma: "LUMA", luma_inverted: "LUMA_INVERTED" })[L.matte.mode]];
        layer.setTrackMatte(matte, mode);
        if (L.matte.keep_matte_visible) { matte.enabled = true; }
        if (!layer.trackMatteLayer || layer.trackMatteLayer.name !== matte.name) { AE.fail("VERIFY_FAILED", "track matte for '" + L.id + "' did not attach"); }
    }

    // ================================================================ main
    try {
        app.beginSuppressDialogs();
        AE.enableFileAccess();
        AE.assertIsolated(job.token);
        manifest.ae = AE.versionInfo();

        var cs = job.settings.composition;
        var comp = app.project.items.addComp(job.composition, cs.width, cs.height, 1.0, cs.duration_s, cs.fps);
        comp.bgColor = [0, 0, 0];
        var mb = cs.motion_blur;
        comp.motionBlur = !!mb.enabled;
        comp.shutterAngle = mb.shutter_angle;
        comp.shutterPhase = mb.shutter_phase;
        comp.motionBlurSamplesPerFrame = mb.samples_per_frame;
        comp.motionBlurAdaptiveSampleLimit = mb.adaptive_sample_limit;

        var layers = job.settings.layers;
        // Layers are given bottom-up; AE adds new layers on top, so adding in
        // order yields the requested stacking.
        for (var i = 0; i < layers.length; i++) {
            var L = layers[i];
            step(L.id + ": create " + L.kind);
            var layer;
            if (L.kind === "text") { layer = makeText(comp, L); }
            else if (L.kind === "shape") { layer = makeShape(comp, L); }
            else { layer = makeImage(comp, L); }
            step(L.id + ": name");
            layer.name = L.id;
            step(L.id + ": timing"); setTiming(layer, L);
            step(L.id + ": anchor"); resolveAnchor(layer, L);
            step(L.id + ": transform"); applyTransform(layer, L);
            step(L.id + ": effects"); var fxs = addEffects(layer, L);
            step(L.id + ": keyframes"); addKeyframes(layer, L, fxs);
        }
        for (var j = 0; j < layers.length; j++) { if (layers[j].clip) { step(layers[j].id + ": clip"); applyClip(comp, layers[j]); } }
        for (var m = 0; m < layers.length; m++) { if (layers[m].matte) { step(layers[m].id + ": matte"); applyMatte(comp, layers[m]); } }

        step("output module templates");
        var templates = AE.outputModuleTemplates(comp);
        if (!AE.contains(templates, job.om_template)) {
            AE.fail("OM_TEMPLATE_MISSING", "output module template '" + job.om_template + "' is not defined; available: " + templates.join(" | "));
        }
        step("save");
        var out = new File(job.project_path);
        app.project.save(out);
        if (!out.exists) { AE.fail("JSX_ERROR", "project save reported success but " + job.project_path + " is missing"); }
        step("summary");
        finish({ ok: true, project_file: String(out.fsName), comp: AE.compSummary(comp), om_templates: templates, om_template: job.om_template });
    } catch (e) {
        try { finish({ ok: false, error: AE.errorInfo(e, e.aeCode) }); } catch (e2) { }
    } finally {
        try { app.endSuppressDialogs(false); } catch (e3) { }
        try { app.project.dirty = false; } catch (e4) { }
        AE.quitIfIsolated();
    }
})();
