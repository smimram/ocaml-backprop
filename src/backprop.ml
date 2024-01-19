(** Backpropagatable functions. *)

(** The backpropagation "functor": it consists of the result of the evaluation
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
let var ~rate x : 'a t =
  !x, (fun g -> x := !x -. rate *. g)

(** A backpropagatable function from a differentiable one. *)
let of_differentiable (f : ('a,'b) Differentiable.t) : 'a t -> 'b t =
  fun (x,k) ->
  let y, df = f x in
  y, fun d -> k (df d)

(** Sigmoid. *)
let sigmoid = of_differentiable Differentiable.sigmoid

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

  (** Add a bias vector which can be optimized. *)
  let bias ~rate b : Vector.t t -> Vector.t t =
    fun (x,k) ->
    Vector.add x !b,
    fun d -> b := Vector.sub !b (Vector.cmul rate d); k d

  (** Apply a linear transformation. *)
  let linear ~rate w : Vector.t t -> Vector.t t =
    fun (x,k) ->
    Vector.Matrix.app !w x,
    fun d -> w := Vector.Matrix.mapi (fun j i w -> w -. rate *. d.(j) *. x.(i)) !w; k (Vector.Matrix.tapp !w d)

  (** Affine layer. *)
  let affine ~rate w b x = x |> linear ~rate w |> bias ~rate b

  (** Sigmoid layer. *)
  let sigmoid = of_differentiable Differentiable.Vector.sigmoid

  let activation kind =
    match kind with
    | `None -> Fun.id
    | `Sigmoid -> sigmoid

  let bias_fun = bias
  let activation_fun = activation

  (** Neural network layer. *)
  let neural_network ?(activation=`Sigmoid) ~weights ?bias ~rate x =
    let x = linear ~rate weights x in
    let x =
      match bias with
      | None -> x
      | Some bias -> bias_fun ~rate bias x
    in
    activation_fun activation x

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
