open Extlib

(** The backpropagation "monad": it consists of the result of the evaluation
    (the primal) and a function to perform the backpropagation given the
    gradient. *)
type 'a t = 'a * ('a -> unit)

(** Backpropagating arrow. *)
type ('a, 'b) arr = ('a -> 'b * ('b -> 'a))

(** A pure value. *)
let return x : 'a t = (x, ignore)

(** Apply a function to an argument. *)
let map (f : ('a, 'b) arr) (x : 'a t) : 'b t =
  let x, k = x in
  let y, l = f x in
  y, (fun v -> k (l v))

let (@@) = map

(** Apply a function to an argument. *)
let (|>) x f = map f x

(** Observe a value being evaluated. *)
let observe f : ('a, 'a) arr =
  fun x -> f x; x, fun d -> d

(** Observe a value being optimized. *)
let observe_descent f : ('a, 'a) arr =
  fun x -> x, fun d -> f d; d

(** Evaluate the result of a computation. *)
let eval (x : 'a t) = fst x

(** Perform gradient descent. *)
let descent (x : float t) = snd x 1.

(** {2 Building blocks} *)

(** A optimized variable. *)
let var rate x : 'a t =
  !x, (fun g -> x := !x -. rate *. g)

(** Sigmoid. *)
let sigmoid : (float, float) arr =
  fun x ->
  let y = sigmoid x in
  y, fun d -> d *. y *. (1. -. y)

let relu : (float, float) arr =
  fun x ->
  relu x, fun d -> d *. step x

(** Sine. *)
let sin : (float, float) arr =
  fun x -> sin x, fun d -> d *. cos x

module Array = struct
  let map (f : ('a, 'b) arr) : ('a array, 'b array) arr =
    fun a ->
      let a = Array.map (map f) a in
      Array.map fst a,
end
