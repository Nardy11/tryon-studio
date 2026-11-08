# Safety and privacy rules for integrating the try-on

For the team connecting the try-on server to the app. The server enforces everything below on its
side; the app must not undo it.

## What the server guarantees

**Every photo is checked twice before the GPU runs.**

1. *Safety:* the person and the outfit are dressed. Nudity, underwear, swimwear and see-through
   clothes are refused. Children's photos are allowed, and their results are checked more
   strictly.
2. *Quality:* the photo is clear enough to give a good result. Photos that are too small, blurry,
   too dark or too bright are refused, as are photos with no face, more than one person, the
   person too far away, or no shoulders and hips visible. Bottoms and full outfits also need the
   legs. A refused photo comes back as HTTP 422 with a message asking for another photo. No GPU
   time is spent on it.

**Every result is checked three ways before anyone can see it:**

- a nudity classifier;
- a garment classifier (underwear, swimwear, see-through, nudity);
- a skin-coverage check per body region against the outfit photo. Skin may show only where the
  outfit itself shows skin.

A result that fails is generated once more. If it fails again, it is deleted and the job ends with
"We couldn't make this look properly."

**Nothing is stored.**

- Photos live only in the server's memory while the look is being made, then are dropped.
- The result is handed out once (`GET /api/jobs/{id}/result`) and deleted the moment it is
  downloaded. If it is never collected, it is deleted after 2 minutes (`TRYON_RESULT_TTL_S`).
- No database, no disk, no bucket on the server side.
- Exception, developer machines only: `TRYON_DEBUG_DIR` writes every photo to disk for tuning,
  and the server logs a warning when it is on. It must never be set in production.

**It fails closed.** If the safety models can't run, try-on stops with "try-on is paused". Nothing
is ever generated unchecked.

## What the app must do

1. **Never store the user's photo or the result** in the database, S3 or any cache. Send the photo
   to the try-on server, show the result, and let it go. If the user wants to keep it, they
   download it to their own device.
2. **If S3 must be used in the middle** (for example, mobile uploads go to a bucket first):
   - use a private bucket with a lifecycle rule that deletes objects within an hour;
   - delete the object as soon as the try-on server has it;
   - never show or share an S3 link to a photo that has not passed the server's checks.
3. **Show only what the server returns.** Don't render the uploaded outfit onto the person in the
   app. The removed 2D preview did that, and it skipped every check.
4. **Show the server's messages as they are.** "Please upload another photo…" asks for a retake.
   "We couldn't make this look properly." is the rare model failure.
5. **No free text to the model.** The server ignores any description. Don't add a text box that
   reaches the prompt.
6. **Put the server behind the app's login and a rate limit.** It has no accounts of its own.

## Known limits

- The refusal of real underwear, swimwear and nude photos has only been tested with simulated
  images (skin painted over clothes). Run a licensed test set through it before launch and confirm
  every image is refused.
- The child detector is approximate. It only tightens the coverage check; it refuses nothing.
- A photo of someone else can be uploaded. Consider a "this is me, or I have their permission"
  confirmation in the app.
