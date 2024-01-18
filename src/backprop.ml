open Extlib

(** The backpropagation "monad": it consists of the result of the evaluation
    (the primal) and a function to perform the backpropagation given the
    gradient. *)
type 'a t = 'a * ('a -> unit)

(** Observe a value being evaluated. *)
let observe f : 'a t -> 'a t =
  fun (x,k) -> (f x; x), k

(** Observe a value being optimized. *)
let observe_descent f : 'a t -> 'a t =
  fun (x,k) -> x, fun d -> f d; k d

(** Evaluate the result of a computation. *)
let eval (x : 'a t) = fst x

(** Perform gradient descent. *)
let descent (x : float t) = snd x 1.

(** {2 Building blocks} *)

(** A optimized variable. *)
let var rate x : 'a t =
  !x, (fun g -> x := !x -. rate *. g)

(** Sigmoid. *)
let sigmoid : float t -> float t =
  fun (x,k) ->
  let y = sigmoid x in
  y, fun d -> k (d *. y *. (1. -. y))

let relu : float t -> float t =
  fun (x,k) ->
  relu x, fun d -> k (d *. step x)

(** Sine. *)
let sin : float t -> float t =
  fun (x,k) -> sin x, fun d -> k (d *. cos x)

module Array = struct
  (* let map (f : 'a t -> 'b t) : ('a array) t -> ('b array) t = *)
    (* fun (a,k) -> _ *)
end
