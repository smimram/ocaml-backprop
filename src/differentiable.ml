(** Differentiable functions. *)

open Extlib

(** Derivable functions. *)
module Derivable = struct
  type t = (float -> float) * (float -> float)

  let sigmoid = sigmoid, fun x -> sigmoid x *. (1. -. sigmoid x)

  let relu = relu , step

  let sin = sin , cos
end

(** A differentiable function. *)
type ('a, 'b) t = 'a -> ('b * ('b -> 'a))

let of_derivable ((f, f'):Derivable.t) : (float, float) t =
  fun x -> f x, fun d -> d *. f' x

type scalar = (float , float) t

type vector = (Vector.t , Vector.t) t

let sigmoid : scalar = of_derivable Derivable.sigmoid

let relu : scalar = of_derivable Derivable.relu

let sin : scalar = of_derivable Derivable.sin
