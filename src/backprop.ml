open Extlib

(** The backpropagation "monad": it consists of the result of the evaluation
    (the primal) and a function to perform the backpropagation given the
    gradient. *)
type 'a t = 'a * ('a -> unit)

type 'a backprop = 'a t

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

(** ReLU. *)
let relu : float t -> float t =
  fun (x,k) ->
  relu x, fun d -> k (d *. step x)

(** Sine. *)
let sin ((x,k) : float t) : float t =
  sin x, fun d -> k (d *. cos x)

(** Square. *)
let square ((x,k) : float t) : float t =
  x *. x, fun d -> k (d *. 2. *. x)

(** Apply a pair of functions to a pair. *)
let map_pair (f : 'a t -> 'c t) (g : 'b t -> 'd t) : ('a * 'b) t -> ('c * 'd) t =
  fun ((a,b),k) ->
  let ar' = ref None in
  let c, kf = f (a, fun a' -> ar' := Some a') in
  let d, kg = g (b, fun b' -> k (Option.get !ar', b')) in
  (c,d),(fun (c',d') -> kf c'; kg d')

module Vector = struct
  type nonrec t = Vector.t t

  (** Squared norm. *)
  let squared_norm ((x,k) : t) : float backprop =
    Vector.squared_norm x, fun d -> k (Vector.cmul (2. *. d) x)
end
