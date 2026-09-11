/*
 * Reopen a saved project and report its compositions and layers, so the
 * driver can confirm the delivered .aep still holds editable text layers.
 *
 * Input: $.global.AE_JOB = { manifest_path, project_path, composition }.
 * Read-only: the project is opened, summarised, closed without saving.
 */
(function () {
    var job = $.global.AE_JOB;
    var manifest = { ok: false, stage: "inspect", started_at: (new Date()).toString() };
    try {
        app.beginSuppressDialogs();
        AE.enableFileAccess();
        AE.assertIsolated(job.token);
        manifest.ae = AE.versionInfo();
        var f = new File(job.project_path);
        if (!f.exists) { AE.fail("ASSET_MISSING", "project not found: " + job.project_path); }
        app.open(f);
        var comp = AE.findComp(job.composition);
        if (!comp) { AE.fail("VERIFY_FAILED", "composition '" + job.composition + "' not found in " + job.project_path); }
        manifest.ok = true;
        manifest.comp = AE.compSummary(comp);
        var comps = [];
        for (var i = 1; i <= app.project.numItems; i++) {
            var it = app.project.item(i);
            if (it instanceof CompItem) { comps.push(String(it.name)); }
        }
        manifest.compositions = comps;
        AE.writeManifest(job.manifest_path, manifest);
    } catch (e) {
        try { manifest.error = AE.errorInfo(e, e.aeCode); AE.writeManifest(job.manifest_path, manifest); } catch (e2) { }
    } finally {
        try { app.endSuppressDialogs(false); } catch (e3) { }
        try { app.project.close(CloseOptions.DO_NOT_SAVE_CHANGES); } catch (e4) { }
        AE.quitIfIsolated();
    }
})();
