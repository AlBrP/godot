extends RigidBody2D

var density = 0
var pressure=Vector2(0,0)
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	#set_process(false)
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
func _integrate_forces(state: PhysicsDirectBodyState2D) -> void:
	state.apply_central_force(pressure)