# pystk-gym
A gym env for the game SuperTuxKart using pystk.

Here is what I would like to do for a VAE:

* don't calculate info (waste of computation)
* ability to save photos locally
* human controllable kart and env

I should be able to hook up a human controller to the env. check what methods i need to change

One idea i have now is to:

* limit it to 1 kart if it's a human controllable env
* init env with a plt window and on render function call, I would render the image to the plt
  window and on step i would get the actions from the plt window. What would be a better way to get
  the actions as a param instead of calculationg it myself inside the env (take a look at
  highway_env)

TODO:

* should i add enum type for all the infos?


BIG TODO:

* The action spaces are not compatible with the gym specs
* Same goes with the observation spaces
* Gym specs don't support/encourage returning multiple observations from a single env, ig that is what VecEnv is supposed to do, but in our case that would mean waste of resources.
* So this obviously fails the `check_env` test from `stable_baselines3`.
* i could prolly make this env "work" like a VecEnv (in a way that would trick gym) so that it's compatible but why so many complications?
