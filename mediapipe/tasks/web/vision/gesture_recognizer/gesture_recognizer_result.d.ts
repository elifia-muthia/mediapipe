/**
 * Copyright 2022 The MediaPipe Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {Category} from '../../../../tasks/web/components/containers/category';
import {Landmark, NormalizedLandmark} from '../../../../tasks/web/components/containers/landmark';

export {Category, Landmark, NormalizedLandmark};

/**
 * Represents the gesture recognition results generated by `GestureRecognizer`.
 */
export declare interface GestureRecognizerResult {
  /** Hand landmarks of detected hands. */
  landmarks: NormalizedLandmark[][];

  /** Hand landmarks in world coordinates of detected hands. */
  worldLandmarks: Landmark[][];

  /** Handedness of detected hands. */
  handedness: Category[][];

  /**
   * Handedness of detected hands.
   * @deprecated Use `.handedness` instead.
   */
  handednesses: Category[][];

  /**
   * Recognized hand gestures of detected hands. Note that the index of the
   * gesture is always -1, because the raw indices from multiple gesture
   * classifiers cannot consolidate to a meaningful index.
   */
  gestures: Category[][];
}
