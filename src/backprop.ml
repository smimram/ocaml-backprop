(** The backpropagation "monad": it consists of the result of the evaluation
    (the primal) and a function to perform the backpropagation given the
    gradient. *)
type 'a t = 'a * ('a -> unit)

(** Backpropagating arrow. *)
type ('a, 'b) arr = ('a -> 'b * ('b -> 'a))

(** A pure value. *)
let return x : 'a t = (x, ignore)

let observe f : ('a, 'a) arr =
  fun x -> f x; x, fun d -> d

let observe_descent f : ('a, 'a) arr =
  fun x -> x, fun d -> f d; d

let app (x : 'a t) (f : ('a, 'b) arr) : 'b t =
  let x, k = x in
  let y, l = f x in
  y, (fun v -> k (l v))

let (|>) = app

let eval (x : 'a t) = fst x

(** A optimized variable. *)
let var rate x : 'a t =
  !x, (fun g -> x := !x -. rate *. g)

let sin : (float, float) arr =
  fun x -> sin x, fun dy -> cos x *. dy

(** Perform gradient descent. *)
let descent (x : float t) = snd x 1.
