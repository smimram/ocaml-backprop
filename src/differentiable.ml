(** Differentiable functions. *)

open Algebra

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

  (** The log function. *)
  let log : t = log10, fun x -> (log10 @@ exp 1.) /. x
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

(** The log function. *)
let log = of_derivable Derivable.log

(*
(** Functions operating on pairs. *)
module Product = struct
  let unit_left : ((unit * 'a) , 'a) t =
    fun ((), x) -> x, fun x' -> (), x'

  let unit_right : (('a * unit) , 'a) t =
    fun (x, ()) -> x, fun x' -> (), x'
end
*)

module Linear = struct
  (** Apply a linear function to a vector. *)
  let app : (Matrix.t * Vector.t, Vector.t) t =
    fun (m, x) -> Linear.app m x, fun d -> Linear.mapi (fun i j _ -> x.(i) *. d.(j)) m, Matrix.tapp m d
end

(** Functions operating on vectors. *)
module Vector = struct
  let to_scalar : (Vector.t, float) t =
    fun x -> Vector.to_scalar x, fun d -> Array.make 1 d

  (** Add a constant. *)
  let cadd a : (Vector.t, Vector.t) t =
    fun x -> Vector.cadd a x, fun d -> d

  (** Multiply by a constant. *)
  let cmul a : (Vector.t, Vector.t) t =
    fun x -> Vector.cmul a x, fun d -> Vector.cmul a d

  (** Add two vectors. *)
  let add : (Vector.t * Vector.t, Vector.t) t =
    fun (x, y) -> Vector.add x y, fun d -> d, d

  (** Hadamrd product of two vectors. *)
  let hadamard : (Vector.t * Vector.t, Vector.t) t =
    fun (x, y) -> Vector.hadamard x y, fun d -> Vector.hadamard d y, Vector.hadamard d x

  (** Sum. *)
  let sum : (Vector.t, float) t = 
    fun x -> Vector.sum x, fun d -> Vector.fill (Vector.dim x) d

  (** Dot product. *)
  let dot : (Vector.t * Vector.t, float) t = 
    fun (x,y) -> Vector.dot x y, fun d -> Vector.cmul d y, Vector.cmul d x

  (*
  let softmax : (Vector.t, Vector.t) t =
    fun x ->
    let s = Vector.softmax x in
    s, fun d ->
      let h = Vector.hadamard s d in
      let b = Vector.sum h in
      Vector.sub h
      si di - sum j si sj dj
  *)

  (** Squared norm of a vector. *)
  let squared_norm : (Vector.t, float) t =
    fun x -> Vector.squared_norm x, fun d -> Vector.cmul (2. *. d) x

  (** Squared distance to a fixed vector. *)
  let squared_distance_to x0 : (Vector.t, float) t =
    fun x ->
    let diff = Vector.sub x x0 in
    Vector.squared_norm diff, fun d -> Vector.cmul (2. *. d) diff
  
  (** Softmax function. *)
  (* There should be an optimisation when composed with crossentropy*)
  let softmax : (Vector.t, Vector.t) t =
    fun x -> (
      let s = Vector.softmax x in
      let n = Vector.dim x in
      let jac = Matrix.init n n
        (fun i j ->
          if i = j then
            s.(i)*.(1.-.s.(i))
          else
            -. s.(i)*.s.(j)
          ) in
      s,
      fun d -> Matrix.app jac d)

  (** Pointwise application of a differentiable function. *)
  let map (f : (float, float) t) : (Vector.t, Vector.t) t =
    fun x ->
      let y = Array.map f x in
      Array.map fst y,
      fun d -> Array.map2 (fun (_,f) d -> f d) y d

  (** Pointwise sigmoid. *)
  let sigmoid = map sigmoid

  (** Pointwise hyperbolic tangent. *)
  let tanh = map tanh

  (** Pointwise rectified linear unit. *)
  let relu = map relu

  (** Pointwise log. *)
  let log = map log
end

module Matrix = struct
  (** Convolve the second matrix by the first one (supposed to be smaller). *)
  let convolution : (Matrix.t * Matrix.t, Matrix.t) t =
    fun (k,x) ->
      Matrix.init
        (Matrix.rows x + 1 - Matrix.rows k)
        (Matrix.cols x + 1 - Matrix.cols k)
        (fun j i ->
           let z = ref 0. in
           for j' = 0 to Matrix.rows k - 1 do
             for i' = 0 to Matrix.cols k - 1 do
               z := !z +. Matrix.get k j' i' *. Matrix.get x (j+j') (i+i')
             done
           done;
           !z
        ), fun d ->
        let dk =
          Matrix.init (Matrix.rows k) (Matrix.cols k)
            (fun j' i' ->
               let z = ref 0. in
               for j = 0 to Matrix.rows x - j' - 1 do
                 for i = 0 to Matrix.cols x - i' - 1 do
                   z := !z +. Matrix.get x (j+j') (i+i') *. Matrix.get d j i
                 done
               done;
               !z
            )
        in
        let dx =
          Matrix.init (Matrix.rows x) (Matrix.cols x)
            (fun q p ->
               let z = ref 0. in
               for j = 0 to min q (Matrix.rows x - 1) do
                 for i = 0 to min p (Matrix.cols x - 1) do
                   z := !z +. Matrix.get k (q-j) (p-i) *. Matrix.get d j i
                 done
               done;
               !z
            )
        in
        dk, dx
end
