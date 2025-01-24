(** Differentiable functions. *)

open Extlib

(** Derivable functions. *)
module Derivable = struct
  (** A derivable function consists of a function together with the associated derivative. *)
  type t = (float -> float) * (float -> float)

  (** Apply a derivable function. *)
  let map ((f,_):t) = f

  (** Derivative of a derivable function. *)
  let derivative ((_,f'):t) = f'

  (** The sigmoid activation function. *)
  let sigmoid : t = sigmoid, fun x -> let y = sigmoid x in y *. (1. -. y)

  (** Hyperbolic tangent. *)
  let tanh : t = tanh, fun x -> let y = tanh x in 1. -. y *. y

  (** The rectified linear unit activation function. *)
  let relu : t = relu , step

  (** The square function. *)
  let square : t = (fun x -> x *. x), (fun x -> 2. *. x)

  (** The sine function. *)
  let sin : t = sin , cos
end

(** A differentiable function. Given an input x, such a function f returns the output f(x), and the function which given the variation in the output provides the variation in the input. *)
type ('a, 'b) t = 'a -> ('b * ('b -> 'a))

(** Sequential composition. *)
let seq : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t =
  fun f g a ->
  let b, f' = f a in
  let c, g' = g b in
  c, fun c' -> f' (g' c')

let ($>) = seq

(** Build from a derivable real function. *)
let of_derivable ((f, f'):Derivable.t) : (float, float) t =
  fun x -> f x, fun d -> d *. f' x

(** The sigmoid activation function. *)
let sigmoid : (float, float) t =
  (* Slightly more efficient implementation than of_derivable Derivable.sigmoid since we compute y once. *)
  fun x -> let y = sigmoid x in y, fun d -> d *. y *. (1. -. y)

(** Hyperbolic tangent. *)
(* TODO: could be optimized like sigmoid. *)
let tanh = of_derivable Derivable.tanh

(** The rectified linear unit activation function. *)
let relu = of_derivable Derivable.relu

(** The square function. *)
let square = of_derivable Derivable.square

(** The sine function. *)
let sin = of_derivable Derivable.sin

(** Functions operating on pairs. *)
module Product = struct
  let unit_left : ((unit * 'a) , 'a) t =
    fun ((), x) -> x, fun x' -> (), x'

  let unit_right : (('a * unit) , 'a) t =
    fun (x, ()) -> x, fun x' -> (), x'
end

(** Functions operating on vectors. *)
module Vector = struct
  (** Squared norm of a vector. *)
  let squared_norm : (Vector.t, float) t =
    fun x -> Vector.squared_norm x, fun d -> Vector.cmul (2. *. d) x

  (** Squared distance to a fixed vector. *)
  let squared_distance_to x0 : (Vector.t, float) t =
    fun x ->
    let diff = Vector.sub x x0 in
    Vector.squared_norm diff, fun d -> Vector.cmul (2. *. d) diff
  
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
