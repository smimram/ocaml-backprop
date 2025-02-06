let vocabulary_size = 12

(** Fischer-Yates shuffle*)
let shuffle arr =
  let n = Array.length arr in
  for i = n - 1 downto 1 do
    let j = Random.int (i + 1) in
    let temp = arr.(i) in
    arr.(i) <- arr.(j);
    arr.(j) <- temp
  done

let generate_sums dataset_size sequence_length max_num =
  
  let x = Array.init dataset_size
    (fun _ -> 
      Array.init sequence_length 
        (fun _ -> Random.int max_num))
  in
  let y = Array.map (Array.fold_left (+) 0) x in
  x,y

let pad n s =
  s ^ String.make (n - String.length s) ' '

let int_lenght n =
  String.length @@ string_of_int n

let generate_string_data dataset_size sequence_length max_num =
  let x, y = generate_sums dataset_size sequence_length max_num in
  let max_size_x = sequence_length - 1 + 
    (int_lenght max_num)*sequence_length in
  let x = Array.map 
    (fun x -> 
      let s = ref @@ string_of_int x.(0) in
      for i = 1 to sequence_length-1 do
        s := !s ^ "+" ^ string_of_int x.(i)
      done;
      pad max_size_x !s
      ) x in
  let max_size_y = int_lenght max_num*sequence_length in
  let y = Array.map 
      (fun x ->
        pad max_size_y @@ string_of_int x
        ) 
      y in
  x,max_size_x,y,max_size_y

let char_to_index c =
  match c with
  | ' ' -> 0
  | '+' -> 1
  | '0'..'9' -> Char.code c - 46
  | _ ->  invalid_arg "Must be a numeral, a space, or a +"

let unit i =
  assert (i < vocabulary_size);
  let a = Array.make vocabulary_size 0. in
  a.(i) <- 1.;
  a

let string_array_to_array_array_array size1 size2 =
  Array.map2 
  (fun s1 s2 -> 
    Array.init size1 (fun n -> unit @@ char_to_index s1.[n]),
    Array.init size2 (fun n -> unit @@ char_to_index s2.[n])
    )

let generate_data dataset_size sequence_length max_num =
  let x,max_x,y,max_y = generate_string_data dataset_size sequence_length max_num in
  string_array_to_array_array_array max_x max_y x y 


  
