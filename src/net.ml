(** Networks consist of backpropagatable functions. *)

open Extlib

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
let update d (x : 'a t) = snd x d

(** Perform gradient climbing. *)
let climb eta x : unit t =
  (), fun () -> update eta x

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
    let make x : E.t t = !x, (fun d -> x := E.add !x d)

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
  (eval x, eval y), fun (d1,d2) -> update d1 x; update d2 y

(** Unpair two values. *)
(* TODO: we should be able to implement this with dup and projections... *)
let unpair (p : ('a * 'b) t) : 'a t * 'b t =
  let x, y = eval p in
  let dl = ref None in
  let dr = ref None in
  (* We only update when we have both values. *)
  let update () =
    match !dl, !dr with
    | Some dl, Some dr -> update (dl, dr) p
    | _ -> ()
  in
  let x = x, fun d -> dl := Some d; update () in
  let y = y, fun d -> dr := Some d; update () in
  x, y

(*
(** Diagonal. *)
let diag (x : 'a t) : ('a * 'a) t =
  let y = eval x in
  (y, y), fun (d1,d2) -> update (d1 +. d2) x

(** Values of this type are {i linear}. This operator must be used in order to use a value twice. *)
let dup x = unpair @@ diag x
*)

let dup n x =
  let dr = ref 0. in
  let counter = ref n in
  let update d =
    decr counter;
    assert (!counter >= 0);
    dr := !dr +. d;
    if !counter = 0 then update !dr x;
  in
  eval x, update

(** Values of this type are {i linear}. This operator must be used when a value is never used. *)
let drop x = update 0. x

let mux (x : ('a t) array) : 'a array t =
  Array.map eval x, fun d -> Array.iter2 update d x

let demux (p : 'a array t) : 'a t array =
  let x = eval p in
  let n = Array.length x in
  let k = ref 0 in
  let dd = Array.make n 0. in
  let update () =
    incr k;
    assert (!k <= n);
    if !k = n then (update dd p)
  in
  Array.mapi (fun i x -> x, (fun d -> dd.(i) <- d; update ())) x

(** Linear transformations. *)
module Linear = struct
  include VarMake (Vector.Linear)

  let app f x = of_differentiable Differentiable.Linear.app @@ pair f x
end

(** Operations on vectors. *)
module Vector = struct
  include VarMake(Vector)

  (*
  (* TODO: we could have a functor to share this definition with above *)
  let diag x : (Vector.t * Vector.t) t =
    let y = eval x in
    (y, y), fun (d1,d2) -> update (Vector.add d1 d2) x

  let dup x = unpair @@ diag x
  *)

  (** Make a value useable exactly n times. *)
  let dup ?label n x =
    let dr = ref @@ Vector.zero @@ Vector.dim (eval x) in
    let counter = ref n in
    let update d =
      decr counter;
      if !counter < 0 then failwith "dup %s used more than %d times" (Option.value ~default:"<unknown>" label) n;
      dr := Vector.add !dr d;
      if !counter = 0 then update !dr x;
    in
    eval x, update

  (** Should be called when a vector is not used. *)
  let drop x = update (Vector.zero (Array.length (eval x))) x

  (** Drop a pair of vectors. *)
  let drop_pair x =
    let v1, v2 = eval x in
    let z1 = Vector.zero (Array.length v1) in
    let z2 = Vector.zero (Array.length v2) in
    update (z1, z2) x

  let to_scalar = of_differentiable Differentiable.Vector.to_scalar

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

  (** Affine layer. *)
  let affine w b x = x |> Linear.app w |> bias b

  (** Sigmoid layer. *)
  let sigmoid = of_differentiable Differentiable.Vector.sigmoid

  let tanh = of_differentiable Differentiable.Vector.tanh

  (** Rectified linear unit. *)
  let relu = of_differentiable Differentiable.Vector.relu

  let activation kind =
    match kind with
    | `None -> Fun.id
    | `ReLU -> relu
    | `Tanh -> tanh
    | `Sigmoid -> sigmoid

  let bias_fun = bias
  let activation_fun = activation

  (** Neural network layer. *)
  let neural_network ?(activation=`Sigmoid) ~weights ?bias x =
    let x = Linear.app (Linear.var weights) x in
    let x =
      match bias with
      | None -> x
      | Some bias ->
        assert (Vector.dim !bias = Vector.Linear.tgt !weights);
        bias_fun bias x
    in
    activation_fun activation x

  (** Recurrent neural network. *)
  module RNN = struct
    (** A {{:https://en.wikipedia.org/wiki/Recurrent_neural_network}recurrent neural network} takes a state and a value and returns a new state and a new value. *)
    type ('s, 'a, 'b) rnn = 's t -> 'a t -> 's t * 'b t

    (** {{:https://en.wikipedia.org/wiki/Gated_recurrent_unit}Gated recurrent unit} layer. The argument is the state and then the input. *)
    let gated_recurrent_unit ~state_weight ~weight ~bias : (Vector.t, Vector.t, Vector.t) rnn =
      fun s x ->
      let wz, wr, wh = weight in
      let uz, ur, uh = state_weight in
      let bz, br, bh = bias in
      let wz = Linear.var wz in
      let wr = Linear.var wr in
      let wh = Linear.var wh in
      let uz = Linear.var uz in
      let ur = Linear.var ur in
      let uh = Linear.var uh in
      let bz = var bz in
      let br = var br in
      let bh = var bh in
      let x = dup 3 x in
      let s = dup 4 s in
      let z = dup 2 @@ sigmoid @@ add (add (Linear.app wz x) (Linear.app uz s)) bz in
      let r = sigmoid @@ add (add (Linear.app wr x) (Linear.app ur s)) br in
      let h = tanh @@ add (add (Linear.app wh x) (Linear.app uh (hadamard r s))) bh in
      let y = add (hadamard (cadd 1. (cmul (-1.) z)) s) (hadamard z h) in
      let y = dup 2 y in
      y, y

    (** {{:https://en.wikipedia.org/wiki/Long_short-term_memory}Long short-term memory} or LSTM layer. In terms of dimensios, the state weights are from hidden to hidden, the weights are from inputs to hidden, and the bias are for hidden. *)
    let long_short_term_memory ~weight_state ~weight ~bias : (Vector.t * Vector.t, Vector.t, Vector.t) rnn =
      fun ch x ->
      let c, h = unpair ch in
      let wf, wi, wo, wc = weight in
      let uf, ui, uo, uc = weight_state in
      let bf, bi, bo, bc = bias in
      let ni = Vector.Linear.src !wf in
      let nh = Vector.Linear.tgt !wf in
      assert (Vector.Linear.src !wi = ni);
      assert (Vector.Linear.src !wo = ni);
      assert (Vector.Linear.src !wc = ni);
      assert (Vector.Linear.tgt !wi = nh);
      assert (Vector.Linear.tgt !wo = nh);
      assert (Vector.Linear.tgt !wc = nh);
      assert (Vector.Linear.src !uf = nh);
      assert (Vector.Linear.src !ui = nh);
      assert (Vector.Linear.src !uo = nh);
      assert (Vector.Linear.src !uc = nh);
      assert (Vector.Linear.tgt !uf = nh);
      assert (Vector.Linear.tgt !ui = nh);
      assert (Vector.Linear.tgt !uo = nh);
      assert (Vector.Linear.tgt !uc = nh);
      assert (Vector.dim !bf = nh);
      assert (Vector.dim !bi = nh);
      assert (Vector.dim !bo = nh);
      assert (Vector.dim !bc = nh);
      let wf = Linear.var wf in
      let wi = Linear.var wi in
      let wo = Linear.var wo in
      let wc = Linear.var wc in
      let uf = Linear.var uf in
      let ui = Linear.var ui in
      let uo = Linear.var uo in
      let uc = Linear.var uc in
      let bf = var bf in
      let bi = var bi in
      let bo = var bo in
      let bc = var bc in
      let x = dup ~label:"lstm_x" 4 x in
      let h = dup ~label:"lstm_h" 4 h in
      (* Forget *)
      let f = sigmoid @@ add (add (Linear.app wf x) (Linear.app uf h)) bf in
      (* Input *)
      let i = sigmoid @@ add (add (Linear.app wi x) (Linear.app ui h)) bi in
      (* Output *)
      let o = sigmoid @@ add (add (Linear.app wo x) (Linear.app uo h)) bo in
      (* Candidate cell state. *)
      let c' = sigmoid @@ add (add (Linear.app wc x) (Linear.app uc h)) bc in
      (* Cell state. *)
      let c = dup 2 @@ add (hadamard f c) (hadamard i c') in
      (* Hidden state. *)
      let h = hadamard o (sigmoid c) in
      let h = dup 2 h in
      let ch = pair c h in
      ch, h

    (** Unfold an RNN so that updating is done after n steps. *)
    let unfold (f : (Vector.t, Vector.t, Vector.t) rnn) (s0 : Vector.t t) (x : Vector.t t list) =
      let f s x = fst (f s x) in
      List.fold_left f s0 x

    (** Apply RNN in bulk mode, to an array of input values at once. *)
    let bulk (f : ('s, 'a, 'b) rnn) (s0 : 's t) (x : 'a t array) =
      x |> Array.fold_left_map f s0 |> Pair.map_right mux
  end
end
