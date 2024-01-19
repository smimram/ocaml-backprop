type t = float array

let to_string (x:t) =
  x
  |> Array.to_list
  |> List.map string_of_float
  |> String.concat ", "
  |> fun s -> "["^s^"]"

let of_list l : t =
  Array.of_list l

let dim (x:t) = Array.length x

let scalar x : t = [|x|]

let to_scalar (x:t) =
  assert (dim x = 1);
  x.(0)

let map f (x:t) : t = Array.map f x

let mapi f (x:t) : t = Array.mapi f x

let map2 f (x:t) (y:t) : t = Array.map2 f x y

let add (x:t) (y:t) : t = Array.map2 (+.) x y

let sub (x:t) (y:t) : t = Array.map2 (-.) x y

(** Sum of the entries of the vector. *)
let sum (x:t) = Array.fold_left (+.) 0. x

(** Square of the euclidean norm. *)
let squared_norm (x:t) = Array.fold_left (fun s x -> s +. x *. x) 0. x

let cmul a x = map (fun x -> a *. x) x

let hadamard x y = map2 ( *. ) x y

let init n f : t = Array.init n f

(** Create a uniformly distributed random vector. *)
let uniform min max n =
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

(** Operations on matrices. *)
module Matrix = struct
  (** A matrix. *)
  type nonrec t =
    {
      rows : int; (** Number of rows. *)
      cols : int; (** Number of columns. *)
      vector : t; (** Underlying vector. *)
    }

  let cols a = a.cols

  let rows a = a.rows

  let src a = cols a

  let tgt a = rows a

  (** [get a j i] returns the entry in row [j] and column [i]. *)
  let get a j i = a.vector.(j*a.cols+i)

  (** Apply a matrix to a vector. *)
  let app a x =
    let m = src a in
    let n = tgt a in
    assert (m = dim x);
    init n
      (fun j ->
         let s = ref 0. in
         for i = 0 to m - 1 do
           s := !s +. get a j i *. x.(i)
         done;
         !s)
  
  (** Initialize a matrix. *)
  let init rows cols f =
    (* TODO: more efficient / imperative *)
    let vector = init (rows * cols) (fun k -> f (k / cols) (k mod cols)) in
    { rows; cols; vector }

  let uniform min max rows cols =
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
      vector = map f a.vector;
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
      vector = map2 f a.vector b.vector;
    }
end
