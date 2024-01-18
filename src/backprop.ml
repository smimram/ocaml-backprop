(** The backpropagation "monad": it consists of the result of the evaluation
    (the primal) and a function to perform the backpropagation given the
    gradient. *)
type 'a t = 'a * ('a -> unit)

(** Backpropagating arrow. *)
type ('a, 'b) arr = ('a -> 'b * ('b -> 'a))

(** A pure value. *)
let return x : 'a t = (x, ignore)

let bind (x : 'a t) (f : ('a, 'b) arr) : 'b t =
  let x, k = x in
  let y, l = f x in
  y, (fun v -> k (l v))

let ( let* ) = bind

let eval (x : 'a t) = fst x

let descent (x : float t) = snd x 1.

(** A optimized variable. *)
let var rate x : 'a t =
  !x, (fun g -> x := !x -. rate *. g)

let sin : (float, float) arr =
  fun x -> sin x, fun dy -> cos x *. dy
