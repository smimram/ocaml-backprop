(** The sigmoid activation function. *)
let sigmoid x = 1. /. (1. +. (exp (-.x)))

(** Rectified linear unit. *)
let relu x = max 0. x

(** Step function. *)
let step x = if x <= 0. then 0. else 1.

module Pair = struct
  let map (f, g) (x1,x2) = (f x1, g x2)
end
