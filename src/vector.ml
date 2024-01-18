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

let squared_norm (x:t) = Array.fold_left (fun s x -> s +. x *. x) 0. x

let cmul a x = map (fun x -> a *. x) x

let hadamard x y = map2 ( *. ) x y

let init n f : t = Array.init n f

let copy (x:t) : t = Array.copy x

(** Maximum entry of a vector. *)
let max (x:t) =
  Array.fold_left max (-. infinity) x

let softmax (x:t) : t =
  let x =
    (* Improve numerical stability. *)
    let m = max x in
    map (fun x -> x -. m) x
  in
  let s = map exp x |> sum in
  map (fun x -> exp x /. s) x
