# Proton e.MAS 5 â€” feature video

Editable Remotion reconstruction of the supplied vertical reference. The composition preserves its scene flow: hook, vehicle reveal, animated road drive, seven feature cards, range card, and blue brand outro.

## Main composition

- ID: `ProtonEmas5Showroom`
- Format: 1080 Ã— 1920, 30 fps
- Duration: 802 frames / 26.73 seconds

The supplied soundtrack is slowed to 85.5% to cover the readability-focused extension, with render-time pitch correction. The side-profile driving shot uses two rim overlays derived directly from `public/emas5/side.png`; no generated or substitute vehicle imagery is used.

## Product copy used

- Up to 325 km WLTP range
- DC charging from 30% to 80% in 21 minutes
- 14.6-inch FHD display head unit
- 375-litre boot
- 70-litre frunk
- Rear-wheel drive
- Six airbags and ADAS

Claims were checked against Proton's official e.MAS 5 product page and brochure:

- https://emas.proton.com/e-mas-5/
- https://emas.proton.com/wp-content/uploads/2026/01/PROTON-e.MAS-5-e-Brochure.pdf

## Preview

```powershell
npm run studio
```

Open `http://localhost:3000/ProtonEmas5Showroom`.

The supplied reference MP4 is used only as the soundtrack layer; every visible frame is rebuilt in Remotion with the supplied Proton e.MAS 5 images.
