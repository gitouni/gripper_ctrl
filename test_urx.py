
import urx
import logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)

    rob = urx.Robot("10.40.101.100")
    #rob = urx.Robot("localhost")
    rob.set_tcp((0,0,0,0,0,0))
    rob.set_payload(0.5, (0,0,0))
    try:
        l = 0.05
        v = 0.05
        a = 0.3
        pose = rob.getl()
        print("robot tcp is at: ", pose)
        # P(0.618, 0.045, 0.088) RV(-1.233, 1.246, -1.233)
        # <PoseVec: P(0.618, 0.045, 0.189) RV(-1.232, 1.246, -1.233)
        pose = [0.7705563603386045, 0.09672636812148871, 0.103, 0.04999123874776192, -2.2031353402180303, 2.2181953222678916]
        # pose = (0.618, 0.045, 0.065,-1.232, 1.246, -1.233)
        # rob.movel(pose, acc=a, vel=v)
        rob.movel(pose, acc=a, vel=v)
        # pose = rob.getl()
        # print("robot tcp is at: ", pose)
        # print("absolute move in base coordinate ")
        # pose[2] += l
        # rob.movel(pose, acc=a, vel=v)
        # print("relative move in base coordinate ")
        # rob.translate((0, 0, -l), acc=a, vel=v)
        # print("relative move back and forth in tool coordinate")
        # rob.translate_tool((0, 0, -l), acc=a, vel=v)
        # rob.translate_tool((0, 0, l), acc=a, vel=v)
    finally:
        rob.close()
