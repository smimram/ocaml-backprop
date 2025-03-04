(** The sigmoid activation function. *)
let sigmoid x = 1. /. (1. +. (exp (-.x)))

(** Rectified linear unit. *)
let relu x = max 0. x

(** Step function. *)
let step x = if x <= 0. then 0. else 1.

(** Vectors. *)
module Vector = struct
  (** A vector. *)
  type t = float array

  (** String representation of a vector. *)
  let to_string (x:t) =
    x
    |> Array.to_list
    |> List.map string_of_float
    |> String.concat ", "
    |> fun s -> "["^s^"]"

  let of_list l : t =
    Array.of_list l

  (** Dimension. *)
  let dim (x:t) = Array.length x

  let scalar x : t = [|x|]

  let pair (x,y) : t = [|x;y|]

  let to_scalar (x:t) =
    if dim x <> 1 then failwith (Printf.sprintf "to_scalar: vector of dimension %d instead of 1" (dim x));
    x.(0)

  let map f (x:t) : t = Array.map f x

  let mapi f (x:t) : t = Array.mapi f x

  let map2 f (x:t) (y:t) : t = Array.map2 f x y

  (** Addition. *)
  let add (x:t) (y:t) : t = Array.map2 (+.) x y

  (** Subtraction. *)
  let sub (x:t) (y:t) : t = Array.map2 (-.) x y

  (** Sum of the entries of the vector. *)
  let sum (x:t) = Array.fold_left (+.) 0. x

  (** Dot product. *)
  let dot (x:t) (y:t) = sum @@ map2 ( *.) x y

  (** Square of the euclidean norm. *)
  let squared_norm (x:t) = Array.fold_left (fun s x -> s +. x *. x) 0. x

  (** Add a constant. *)
  let cadd a x = map (fun x -> a +. x) x

  (** Multiply by a constant. *)
  let cmul a x = map (fun x -> a *. x) x

  let hadamard x y = map2 ( *. ) x y

  let init n f : t = Array.init n f

  let fill n x : t = Array.init n (fun _ -> x)

  let zero n = fill n 0.

  (** Create a uniformly distributed random vector. *)
  let uniform ?(min= -1.) ?(max=1.) n =
    let d = max -. min in
    init n (fun _ -> Random.float d +. min)

  let copy (x:t) : t = Array.copy x

  (** Maximum entry of a vector. *)
  let max (x:t) =
    Array.fold_left max (-. infinity) x

  (** Softmax of a vector (useful to convert logits to probabilities). *)
  let softmax (x:t) : t =
    let x =
      (* Improve numerical stability. *)
      let m = max x in
      map (fun x -> x -. m) x
    in
    let s = map exp x |> sum in
    map (fun x -> exp x /. s) x
end

(** Operations on matrices. *)
module Matrix = struct
  (** A matrix. *)
  type nonrec t =
    {
      rows : int; (** Number of rows. *)
      cols : int; (** Number of columns. *)
      vector : Vector.t; (** Underlying vector. *)
    }

  let cols a = a.cols

  let rows a = a.rows

  let src a = cols a

  let tgt a = rows a

  (** [get a j i] returns the entry in row [j] and column [i]. *)
  let get a j i = a.vector.(j*a.cols+i)

  let add a b =
    assert (src a = src b);
    assert (tgt a = tgt b);
    { rows = a.rows; cols = a.cols; vector = Vector.add a.vector b.vector }

  let add_list = function
    | [] -> assert false
    | [x] -> x
    | x::l -> List.fold_left add x l

  let sub a b =
    assert (src a = src b);
    assert (tgt a = tgt b);
    { rows = a.rows; cols = a.cols; vector = Vector.sub a.vector b.vector }

  let cmul x a =
    { rows = a.rows; cols = a.cols; vector = Vector.cmul x a.vector }

  (** Apply a matrix to a vector. *)
  let app a x =
    let m = src a in
    let n = tgt a in
    assert (m = Vector.dim x);
    Vector.init n
      (fun j ->
         let s = ref 0. in
         for i = 0 to m - 1 do
           s := !s +. get a j i *. x.(i)
         done;
         !s)

  (** Initialize a matrix. *)
  let init rows cols f =
    (* TODO: more efficient / imperative *)
    let vector = Vector.init (rows * cols) (fun k -> f (k / cols) (k mod cols)) in
    { rows; cols; vector }

  (** Zero matrix. *)
  let zero rows cols =
    init rows cols (fun _ _ -> 0.)

  (** Random matrix. *)
  let uniform ?(min=(-1.)) ?(max=1.) rows cols =
    let d = max -. min in
    init rows cols (fun _ _ -> Random.float d +. min)

  let transpose a =
    init (cols a) (rows a) (fun j i -> get a i j)

  (** Apply the transpose of a matrix to a vector. *)
  let tapp a x =
    (* TODO: optimize. *)
    app (transpose a) x

  let map f a =
    {
      rows = a.rows;
      cols = a.cols;
      vector = Vector.map f a.vector;
    }

  let mapi f a =
    (* TODO: optimize. *)
    init (rows a) (cols a) (fun j i -> f j i (get a j i))

  (** Map a function to two matrices of the same size. *)
  let map2 f a b =
    assert (src a = src b);
    assert (tgt a = tgt b);
    {
      rows = a.rows;
      cols = a.cols;
      vector = Vector.map2 f a.vector b.vector;
    }
end

(** Linear maps. Those are roughly the same as matrices, excepting for the convention that arguments are input and then output. *)
module Linear = struct
  type t = Matrix.t
  let src (a:t) = Matrix.src a
  let tgt (a:t) = Matrix.tgt a
  let get (a:t) i j = Matrix.get a j i
  let init src tgt f : t = Matrix.init tgt src (fun j i -> f i j)
  let uniform ?min ?max src tgt : t = Matrix.uniform ?min ?max tgt src
  let mapi f (a:t) : t = Matrix.mapi (fun j i w -> f i j w) a
  let add (a:t) (b:t) = Matrix.add a b
  let cmul x a = Matrix.cmul x a

  (** Apply a linear function to a vector. *)
  let app (f:t) x = Matrix.app f x

  (** Apply the transpose of a linear function to a vector. *)
  let tapp (f:t) x = Matrix.tapp f x
end

module Tensor = struct
  (** Dimensions of the tensor. *)
  type dimensions = int list

  type t =
    {
      src : dimensions;
      tgt : dimensions;
      f : Linear.t
    }

  (** Ensure that point x is below the dimension. *)
  let within (dims:dimensions) x =
    let rec aux = function
      | x::xx, d::dims -> x < d && aux (xx,dims)
      | [], [] -> true
      | _ -> false
    in
    aux (x,dims)

  (** Offset of a coordinate. *)
  let offset (dims:dimensions) x =
    assert (within dims x);
    let rec aux p = function
      | x::xx, d::dims -> p * x + aux (p*d) (xx,dims)
      | [], [] -> 0
      | _ -> assert false
    in
    aux 1 (x,dims)

  (** Retrieve an element. *)
  let get a x y =
    Linear.get a.f (offset a.src x) (offset a.tgt y)
end
