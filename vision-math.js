/** Pure geometry/decoding helpers. Coordinates returned here use source-frame pixels. */
export function letterboxTransform(width, height, inputSize) {
  if (![width, height, inputSize].every(Number.isFinite) || Math.min(width, height, inputSize) <= 0) {
    throw new Error('Invalid image dimensions');
  }
  const scale = Math.min(inputSize / width, inputSize / height);
  const resizedWidth = Math.round(width * scale);
  const resizedHeight = Math.round(height * scale);
  return {
    width, height, inputSize, scale, resizedWidth, resizedHeight,
    padLeft: Math.round((inputSize - resizedWidth) / 2 - 0.1),
    padTop: Math.round((inputSize - resizedHeight) / 2 - 0.1),
  };
}

export function unletterboxPoint(x, y, transform, clip = false) {
  let px = (x - transform.padLeft) / transform.scale;
  let py = (y - transform.padTop) / transform.scale;
  if (clip) {
    px = Math.max(0, Math.min(transform.width - 1, px));
    py = Math.max(0, Math.min(transform.height - 1, py));
  }
  return [px, py];
}

export function unletterboxBox(box, transform) {
  const a = unletterboxPoint(box[0], box[1], transform, true);
  const b = unletterboxPoint(box[2], box[3], transform, true);
  return [...a, ...b];
}

export function boxArea(box) {
  return Math.max(0, box[2] - box[0]) * Math.max(0, box[3] - box[1]);
}

export function boxIoU(a, b) {
  const intersection = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]))
    * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  return intersection / Math.max(boxArea(a) + boxArea(b) - intersection, 1e-9);
}

export function nonMaxSuppression(items, threshold = 0.5, limit = 20) {
  const remaining = items.filter(item => Number.isFinite(item.score) && item.box.every(Number.isFinite)
    && boxArea(item.box) > 0).sort((a, b) => b.score - a.score);
  const kept = [];
  for (const item of remaining) {
    if (kept.every(other => boxIoU(item.box, other.box) <= threshold)) kept.push(item);
    if (kept.length >= limit) break;
  }
  return kept;
}

function checkTensor(tensor) {
  if (!tensor?.dims || !tensor?.data || tensor.dims.length !== 3 || tensor.dims[0] !== 1) {
    throw new Error('Unsupported model output; expected batch-one rank-three tensor');
  }
}

export function decodeBody(tensor, transform, threshold = 0.4) {
  checkTensor(tensor);
  const { dims, data } = tensor;
  const items = [];
  if (dims[2] === 57) {
    // YOLO26 end-to-end output: x1,y1,x2,y2,score,class,17*(x,y,confidence).
    for (let row = 0; row < dims[1]; row++) {
      const start = row * 57;
      const score = Number(data[start + 4]);
      if (score < threshold || !Number.isFinite(score) || Math.round(data[start + 5]) !== 0) continue;
      const box = unletterboxBox(Array.from(data.subarray(start, start + 4)), transform);
      const keypoints = [];
      for (let index = 0; index < 17; index++) {
        const offset = start + 6 + index * 3;
        keypoints.push([...unletterboxPoint(data[offset], data[offset + 1], transform), Number(data[offset + 2])]);
      }
      items.push({ box, score, keypoints });
    }
  } else if (dims[1] === 56 || dims[2] === 56) {
    // Optional raw YOLO-pose export: cx,cy,w,h,person_score,17 keypoints.
    const channelFirst = dims[1] === 56;
    const count = channelFirst ? dims[2] : dims[1];
    const at = (channel, row) => Number(data[channelFirst ? channel * count + row : row * 56 + channel]);
    for (let row = 0; row < count; row++) {
      const score = at(4, row);
      if (!Number.isFinite(score) || score < threshold) continue;
      const cx = at(0, row), cy = at(1, row), w = at(2, row), h = at(3, row);
      const keypoints = [];
      for (let index = 0; index < 17; index++) {
        const offset = 5 + index * 3;
        keypoints.push([...unletterboxPoint(at(offset, row), at(offset + 1, row), transform), at(offset + 2, row)]);
      }
      items.push({ box: unletterboxBox([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], transform), score, keypoints });
    }
  } else {
    throw new Error(`Unsupported body output shape: ${dims.join('×')}`);
  }
  return nonMaxSuppression(items, 0.5, 10);
}

export function decodeRackets(tensor, transform, threshold = 0.25) {
  checkTensor(tensor);
  const { dims, data } = tensor;
  const items = [];
  if (dims[1] === 84 || dims[2] === 84) {
    // YOLO11 COCO raw output: xywh + 80 class scores; tennis racket is class 38.
    const channelFirst = dims[1] === 84;
    const count = channelFirst ? dims[2] : dims[1];
    const at = (channel, row) => Number(data[channelFirst ? channel * count + row : row * 84 + channel]);
    for (let row = 0; row < count; row++) {
      const score = at(4 + 38, row);
      if (!Number.isFinite(score) || score < threshold) continue;
      // Match class-aware Ultralytics output: another class must not dominate this box.
      let best = 38;
      for (let cls = 0; cls < 80; cls++) if (at(4 + cls, row) > at(4 + best, row)) best = cls;
      if (best !== 38) continue;
      const cx = at(0, row), cy = at(1, row), w = at(2, row), h = at(3, row);
      items.push({ box: unletterboxBox([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], transform), score });
    }
  } else if (dims[2] === 6) {
    for (let row = 0; row < dims[1]; row++) {
      const offset = row * 6;
      if (data[offset + 4] < threshold || Math.round(data[offset + 5]) !== 38) continue;
      items.push({ box: unletterboxBox(Array.from(data.subarray(offset, offset + 4)), transform), score: Number(data[offset + 4]) });
    }
  } else {
    throw new Error(`Unsupported racket output shape: ${dims.join('×')}`);
  }
  return nonMaxSuppression(items, 0.45, 10);
}

export function selectPerson(people, previousBox = null) {
  if (!people.length) return null;
  if (previousBox) {
    const ranked = people.map(person => ({ person, overlap: boxIoU(person.box, previousBox) }))
      .sort((a, b) => b.overlap - a.overlap);
    if (ranked[0].overlap > 0.15) return ranked[0].person;
  }
  return people.reduce((best, person) => boxArea(person.box) > boxArea(best.box) ? person : best);
}

export function selectRacket(rackets, person, width, height) {
  if (!person?.keypoints) return null;
  const wrists = [person.keypoints[9], person.keypoints[10]]
    .filter(point => point && point.every(Number.isFinite) && point[2] > 0.5);
  if (!wrists.length) return null;
  const ranked = [];
  for (const racket of rackets) {
    const b = racket.box;
    if (!b.every(Number.isFinite) || !Number.isFinite(racket.score)) continue;
    const bw = b[2] - b[0], bh = b[3] - b[1];
    if (Math.min(bw, bh) < 5 || boxArea(b) > width * height * 0.98) continue;
    const size = Math.max(bw, bh);
    const distance = Math.min(...wrists.map(point => Math.hypot(
      Math.max(b[0] - point[0], 0, point[0] - b[2]),
      Math.max(b[1] - point[1], 0, point[1] - b[3]))));
    if (distance < size * 0.5) ranked.push({ racket, rank: distance / size - 0.1 * racket.score });
  }
  ranked.sort((a, b) => a.rank - b.rank);
  return ranked[0]?.racket || null;
}

export function squareCrop(box) {
  const side = Math.max(box[2] - box[0], box[3] - box[1]) * 1.25;
  if (!Number.isFinite(side) || side <= 0) throw new Error('Invalid racket crop');
  return { centerX: (box[0] + box[2]) / 2, centerY: (box[1] + box[3]) / 2, side };
}

export function decodeSimCC(xTensor, yTensor, crop) {
  checkTensor(xTensor); checkTensor(yTensor);
  if (xTensor.dims[1] !== 5 || yTensor.dims[1] !== 5) throw new Error('Expected five racket keypoints');
  const nx = xTensor.dims[2], ny = yTensor.dims[2];
  if (nx !== 512 || ny !== 512) throw new Error('Expected fixed 256×256 / split-ratio-two SimCC output');
  const points = [], scores = [];
  for (let keypoint = 0; keypoint < 5; keypoint++) {
    let ix = 0, iy = 0, sx = -Infinity, sy = -Infinity;
    for (let i = 0; i < nx; i++) {
      const value = Number(xTensor.data[keypoint * nx + i]);
      if (value > sx) { sx = value; ix = i; }
    }
    for (let i = 0; i < ny; i++) {
      const value = Number(yTensor.data[keypoint * ny + i]);
      if (value > sy) { sy = value; iy = i; }
    }
    const score = Math.min(sx, sy);
    scores.push(score);
    points.push(score > 0 && Number.isFinite(score)
      ? [(ix / 2 / 256) * crop.side + crop.centerX - crop.side / 2,
         (iy / 2 / 256) * crop.side + crop.centerY - crop.side / 2]
      : [NaN, NaN]);
  }
  return { points, scores };
}

export function validRacketAxis(points, scores, box, width, height, threshold = 0.5) {
  if (points?.length !== 5 || scores?.length !== 5) return null;
  if (![scores[0], scores[2]].every(Number.isFinite) || Math.min(scores[0], scores[2]) < threshold) return null;
  const a = points[0], b = points[2];
  if (![...a, ...b].every(Number.isFinite) || Math.hypot(a[0] - b[0], a[1] - b[1]) < 8) return null;
  for (const [x, y] of [a, b]) {
    if (!(x >= 0 && x < width && y >= 0 && y < height
      && x >= box[0] - 10 && x <= box[2] + 10 && y >= box[1] - 10 && y <= box[3] + 10)) return null;
  }
  return [a, b];
}
