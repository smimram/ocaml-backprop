(** Differentiable functions. *)

open Extlib

(** Derivable functions. *)
module Derivable = struct
  type t = (float -> float) * (float -> float)

  let sigmoid : t = sigmoid, fun x -> let y = sigmoid x in y *. (1. -. y)

  let relu : t = relu , step

  let square : t = (fun x -> x *. x), (fun x -> 2. *. x)

  let sin : t = sin , cos
end

(** A differentiable function. *)
type ('a, 'b) t = 'a -> ('b * ('b -> 'a))

(** Sequential composition. *)
let seq : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t =
  fun f g a ->
  let b, f' = f a in
  let c, g' = g b in
  c, fun c' -> f' (g' c')

(** Build from a derivable real function. *)
let of_derivable ((f, f'):Derivable.t) : (float, float) t =
  fun x -> f x, fun d -> d *. f' x

let sigmoid : (float, float) t =
  (* Slightly more efficient implementation than of_derivable Derivable.sigmoid
     since we compute y once. *)
  fun x -> let y = sigmoid x in y, fun d -> d *. y *. (1. -. y)

let relu = of_derivable Derivable.relu

let square = of_derivable Derivable.square

let sin = of_derivable Derivable.sin

module Vector = struct
  (** Squared norm of a vector. *)
  let squared_norm : (Vector.t, float) t =
    fun x -> Vector.squared_norm x, fun d -> Vector.cmul (2. *. d) x

  (** Softmax function. *)
  let softmax : (Vector.t, Vector.t) t =
    fun x -> Vector.softmax x, fun _d -> failwith "TODO"

  (** Pointwise application of a differentiable function. *)
  let map (f : (float, float) t) : (Vector.t, Vector.t) t =
    fun x ->
      let y = Array.map f x in
      Array.map fst y,
      fun d -> Array.map2 (fun (_,f) d -> f d) y d

  (** Pointwise sigmoid. *)
  let sigmoid = map sigmoid
end
