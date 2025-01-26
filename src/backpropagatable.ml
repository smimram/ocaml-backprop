(** Backpropagatable functions. *)

(** The backpropagation "functor": it consists of the result of the evaluation (the primal) and a function to perform the backpropagation given the gradient. *)
type 'a t = 'a * ('a -> unit)

(** Observe a value being evaluated. *)
let observe f : 'a t -> 'a t =
  fun (x,k) -> (f x; x), k

(** Observe a value being optimized. *)
let observe_descent f : 'a t -> 'a t =
  fun (x,k) -> x, fun d -> f d; k d

(** Evaluate the result of a computation. *)
let eval (x : 'a t) = fst x

(** Update according to parameter. *)
let update (x : 'a t) d = snd x d

(** Perform gradient climbing. *)
let climb eta x : unit t =
  (), fun () -> update x eta

(** Perform gradient descent. *)
let descent eta = climb (-.eta)

(** Run gradient descent. *)
let run (x : unit t) =
  snd x ()

(** {2 Building blocks} *)

(** A constant. *)
let cst x : 'a t =
  x, (fun _ -> ())

module VarMake (E: sig type t val add : t -> t -> t val cmul : float -> t -> t end) =
struct
  module Var = struct
    let make x : E.t t =
      !x, (fun d -> x := E.add !x d)

    (** A smoothed reference. The first parameter controls smoothing: 1 acts like a traditional reference, 0 never updates. *)
    let smooth a x : E.t t =
      !x, (fun d -> x := E.add !x (E.cmul a d))

    (** A reference which is averaged over n values. *)
    let average n x : E.t t =
      let s = Array.make n !x in
      let i = ref 0 in
      let store y =
        if !i = n then
          (
            let d = ref s.(0) in
            for i = 1 to n - 1 do
              d := E.add !d s.(i)
            done;
            let d = E.cmul (1. /. float n) !d in
            x := E.add !x d;
            i := 0
          );
        s.(!i) <- y;
        incr i
      in
      !x, (fun d -> store d)
  end

  (** A optimized variable. *)
  let var = Var.make
end

include VarMake(struct include Float let cmul = ( *. ) end)

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

(** Fold a function over a series of inputs. *)
let rec fold (f : 'a -> 'b t -> 'b t) (l : 'a Seq.t) (s : 'b t) : 'b t =
  match l () with
  | Nil -> s
  | Cons (x, l) -> f s x |> fold f l

(** Pair two values. *)
let pair (x : 'a t) (y : 'b t) : ('a * 'b) t =
  (eval x, eval y), fun (d1,d2) -> update x d1; update y d2

(** Unpair two values. *)
let unpair (p : ('a * 'b) t) : 'a t * 'b t =
  let x, y = eval p in
  let dl = ref None in
  let dr = ref None in
  (* We only update when we have both values. *)
  let update () =
    match !dl, !dr with
    | Some dl, Some dr -> update p (dl, dr)
    | _ -> ()
  in
  let x = x, fun d -> dl := Some d; update () in
  let y = y, fun d -> dr := Some d; update () in
  x, y

(** Operations on vectors. *)
module Vector = struct
  include VarMake(Vector)

  (** Add a constant. *)
  let cadd a = of_differentiable (Differentiable.Vector.cadd a)

  (** Multiply by a constant. *)
  let cmul a = of_differentiable (Differentiable.Vector.cmul a)

  (** Add two vectors. *)
  let add x y = of_differentiable Differentiable.Vector.add @@ pair x y

  (** Hadamard product of two vectors. *)
  let hadamard x y = of_differentiable Differentiable.Vector.hadamard @@ pair x y

  (** Squared norm. *)
  let squared_norm = of_differentiable Differentiable.Vector.squared_norm

  (** Squared distance to fixed vector. *)
  let squared_distance_to x0 = of_differentiable (Differentiable.Vector.squared_distance_to x0)

  (** Add a bias vector which can be optimized. *)
  let bias b x = add (var b) x

  (** Linear transformations. *)
  module Linear = struct
    include VarMake (Vector.Linear)

    let app f x = of_differentiable Differentiable.Vector.Linear.app @@ pair f x
  end

  (** Affine layer. *)
  let affine w b x = x |> Linear.app w |> bias b

  (** Sigmoid layer. *)
  let sigmoid = of_differentiable Differentiable.Vector.sigmoid

  let tanh = of_differentiable Differentiable.Vector.tanh

  let activation kind =
    match kind with
    | `None -> Fun.id
    | `Sigmoid -> sigmoid

  let bias_fun = bias
  let activation_fun = activation

  (** Neural network layer. *)
  let neural_network ?(activation=`Sigmoid) ~weights ?bias x =
    let x = Linear.app (Linear.var weights) x in
    let x =
      match bias with
      | None -> x
      | Some bias -> bias_fun bias x
    in
    activation_fun activation x

  (** Recurrent neural network. *)
  module RNN = struct
    (** A recurrent neural network takes a state and a value and returns an updated state and a value. *)
    type t = (Vector.t * Vector.t) -> (Vector.t * Vector.t)

    (** {{:https://en.wikipedia.org/wiki/Gated_recurrent_unit}Gated recurrent unit} layer. The argument is the state and then the input. *)
    let gated_recurrent_unit ~weight ~state_weight ~bias (s,x) =
      let wz, wr, wh = weight in
      let uz, ur, uh = state_weight in
      let bz, br, bh = bias in
      let z = sigmoid @@ add (Linear.app wz x) (add (Linear.app uz s) bz) in
      let r = sigmoid @@ add (Linear.app wr x) (add (Linear.app ur s) br) in
      let h = tanh @@ add (Linear.app wh x) (add (Linear.app uh (hadamard r s)) bh) in
      let y = add (hadamard (cadd 1. (cmul (-1.) z)) s) (hadamard z h) in
      y, y
  end
end
