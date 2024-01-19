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

(** A backpropagatable function from a differentiable one. *)
let of_differentiable (f : ('a,'b) Differentiable.t) : 'a t -> 'b t =
  fun (x,k) ->
  let y, df = f x in
  y, fun d -> k (df d)

(** Sigmoid. *)
let sigmoid : float t -> float t =
  fun (x,k) ->
  let y = sigmoid x in
  (* Slightly more efficient than the differentiable implementation since we
     have access to y. *)
  y, fun d -> k (d *. y *. (1. -. y))

(** ReLU. *)
let relu = of_differentiable Differentiable.relu

(** Sine. *)
let sin = of_differentiable Differentiable.sin

(** Square. *)
let square = of_differentiable Differentiable.square

(*
(** Apply a pair of functions to a pair. *)
let map_pair (f : 'a t -> 'c t) (g : 'b t -> 'd t) : ('a * 'b) t -> ('c * 'd) t =
  fun ((a,b),k) ->
  let ar' = ref None in
  let c, kf = f (a, fun a' -> ar' := Some a') in
  let d, kg = g (b, fun b' -> k (Option.get !ar', b')) in
  (c,d),(fun (c',d') -> kf c'; kg d')
*)

(** Operations on vectors. *)
module Vector = struct
  (** Squared norm. *)
  let squared_norm = of_differentiable Differentiable.Vector.squared_norm

  (*
  (** Map a function on a vector. *)
  let map (f : float t -> float t) ((x,k) : vector) : vector =
    let n = Array.length x in
    let r = ref None in
    let k i x' =
      if !r = None then r := Some (Array.make n 0.);
      let r = Option.get !r in
      r.(i) <- x';
      if i = n-1 then k r
    in
    let y = Array.mapi (fun i x -> f (x, k i)) x in
    let y' = Array.map snd y in
    Array.map fst y,
    fun x' -> Array.iteri (fun i k -> k x'.(i)) y'
  *)
end
