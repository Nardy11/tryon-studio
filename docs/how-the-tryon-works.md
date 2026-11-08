# How the try-on works, step by step

P = the person's photo. O = the outfit photo.
Each step says what goes in, which tool does the work, and what comes out.

---

## A. Photo intake and checks

**1. Open the file** (Pillow + pillow-heif)
- In: any image file (JPEG, PNG, WebP, HEIC and more, any size).
- Does: decodes it, fixes the phone's rotation, and shrinks big photos so a 48 MP photo costs about the memory of a 4 MP one.
- Out: an RGB picture, longest side at most 1536 px.

**2. Segmentation: what is each pixel?** (SegFormer clothes parser)
- In: a photo.
- Out: a label for every pixel: background, hair, face, upper clothes, skirt, pants, dress, belt, arms, legs, shoes, bag, scarf.
- Used by nearly everything below: the masks, the colour readings, the face lock and the checks.

**3. Pose: where is the skeleton?** (MediaPipe Pose, CPU)
- In: a photo.
- Out: shoulder, hip, knee and ankle heights.
- We turn these into a **body ruler**: 0 = shoulders, 1 = hips, 2 = knees, 3 = ankles. "Jacket ends at 0.9" means just above the hips on anyone.

**4. Safety check** (two models, CPU)
- **Falconsai ViT:** photo in, "probability this is nude or explicit" out.
- **CLIP ViT-L/14:** we give it sentences such as "a person in lingerie", "a bikini" and "a child", and it scores how well the photo matches each. The unsafe ones are summed.
- Each photo is judged whole and as an upper and a lower crop, and the worst score counts, so a small exposed area can't hide in a big picture.
- Out: pass, "please upload a photo where you are dressed", or "paused" if a model can't run (nothing ever runs unchecked).
- A child is not refused, but it sets a stricter limit for step 12.

**5. Quality check** (OpenCV and the outputs of steps 2 and 3)
- Sharpness: the Laplacian variance of the person's crop. A blurry photo scores under 40 and gets "please upload another photo".
- Brightness: the average grey level on the body. Too dark or too bright is turned away.
- The face labels must make exactly one face, so zero faces or several people are turned away.
- The person must span at least 40% of the photo height, and the skeleton must be found.
- Bottoms and full outfits also need legs in the photo.
- Out: pass, or a plain retake message. Image size alone never fails a photo.

The website runs steps 4 and 5 when you add a photo and blocks **Try it on** until they pass.
The server runs them again on the try-on request itself, so they can't be skipped.

---

## B. Understanding the outfit

**6. Read the pieces** (k-means colour clustering in OpenCV, plus the skeleton)
- In: O, its labels and its skeleton.
- Does:
  - Clusters the clothes pixels by colour.
  - Takes the colour that holds the sides and sleeves as the outer layer, and finds a different-coloured strip down the middle as the top underneath.
  - Measures how far that strip reaches down, to tell open from closed.
  - Reads each piece's lowest edge against the body ruler, giving "hip-length jacket" or "knee-length skirt".
  - Names each colour by nearest match in a colour table.
- Out: a **checklist**, for example: (1) taupe hip-length jacket, closed; (2) dark brown top; (3) dark brown knee-length skirt, each with its measured colour and hem.

**7. Move the hem onto the user's body**
- In: the jacket's ruler position (0.89) and P's skeleton.
- Out: the pixel row where the new jacket should end on P (57% down the photo).

---

## C. Generating

**8. Build the mask: where may the model paint?** (NumPy plus `cv2.dilate`)
- In: P's labels and the hem row.
- Does: takes the old clothes and arms, grows them a little, and extends the area down to the hem row. Face, hair, hands and anything held (a bag) are cut out.
- Out: a black-and-white mask. The crop is set so the whole head is inside it.

**9. Prepare the pictures**
- A crop of P, and a reference crop of O's real pixels, upscaled to 512 px, with the model's face blanked out so the face can't leak.
- The prompt is assembled from fixed words and the checklist, ending with "cover the body exactly as the outfit does, no extra skin".

**10. FLUX.2 Klein** (4-bit, on the GPU)
- In: the crop of P, the reference of O, and the prompt. It runs 4 steps.
- Out: a new image of the person wearing the outfit.

**11. Layer it back onto the original**, bottom to top:
1. P's original photo.
2. Klein's pixels only inside the clothes area (mask plus a margin, feathered by a 2 px blur). The background, face and everything else stay P's pixels.
3. Stray hair Klein drew outside P's real hair is painted over (`cv2.inpaint`).
4. **Checklist test:** the labels are run on the result, and each piece's colour is compared with its checklist colour, relative to what P wore before. If a piece is missing, it generates once more with a new seed and keeps the better result.
5. **Colour fixes** in Lab colour space: each layer pulled back to its true colour, newly drawn kept clothes matched to P's own, and new skin matched to P's skin tone.
6. **Face lock, last and on top:** P's face and hair pixels are pasted over everything, with a soft per-strand edge where hair rests on the shoulders.

---

## D. Checking the result before it can be shown

**12. Three independent checks, all must pass**
1. The Falconsai nudity detector, on the whole result and on crops.
2. CLIP, with the same unsafe-garment sentences as step 4.
3. **Skin by body region** (OpenCV Lab distance to P's own face colour):
   - Counts skin pixels in the chest, belly, thighs and shins, with the garment colours excluded.
   - Compares each region with how much skin O itself shows there, plus a margin (0.15, or 0.10 for a child).
   - Legs below a skirt's own hem are allowed, because that's what that length shows.

If any check fails, the whole generation runs again with a new seed.
If it fails again, the result is deleted and the person sees "We couldn't make this look properly."

---

## E. Delivery

**13.** The result is held in memory and can be downloaded **once**, then it's deleted.
If nobody collects it, it's deleted after 2 minutes.
Photos are dropped as soon as the look is done.
Nothing is written to disk, a database or S3.
Deleting a job also stops its generation on the GPU.

---

## Two extra pieces

- **One look at a time:** the website sends one job at a time and the server has a single GPU worker, which is why the other looks wait their turn.
- **Hand gestures:** MediaPipe Hand Landmarker runs entirely in the browser, with a frame-difference backup. The camera video never leaves the device.

---

## Still open

- The safety checks have been tested on simulated images, not real underwear, swimwear or nude photos. A licensed test set should be run before launch.
- What the app does with photos on its side (database, S3) is outside this server. Rules: `tryon-studio\docs\SAFETY.md`.
- The server has no login or rate limit.
- Accessories can swap (the model's watch instead of the person's), and shoes aren't changed.
