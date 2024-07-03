from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import gui, visual, core, data, logging, parallel
from psychopy.constants import (NOT_STARTED, STARTED, FINISHED)
import os  # handy system and path functions
import psychopy.iohub as io
from psychopy.hardware import keyboard
import time

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2023.1.2'
expName = 'MNS'  
expInfo = {
    'participant':'',
    'Condition (1/2)': ''} # 1 is counting condition; 2 is meaning condition
# --- Show participant info dialog --
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
if expInfo['Condition (1/2)'] != '1' and expInfo['Condition (1/2)'] != '2':
    raise ValueError("Condition must be either '1' or '2'")
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='untitled.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# --- Setup the Window ---
win = visual.Window(
    size=(800, 600), fullscr=True, winType='pyglet', 
    monitor='testMonitor', useFBO=True, units='height')
win.mouseVisible = False
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# --- Setup input devices ---
ioConfig = {}

# Setup iohub keyboard
ioConfig['Keyboard'] = dict(use_keymap='psychopy')

ioServer = io.launchHubServer(window=win, **ioConfig)

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='iohub')

def CHECK_for_QUIT():
    """ Quit the experiment immediately when the 'Esc' key is pressed."""
    if defaultKeyboard.getKeys(keyList=["escape"]):
        win.close()
        core.quit()

def routine_TEXT(TEXT, duration=None, wait_key=None, countdown=None):
    text_stim = visual.TextStim(
        win, text=TEXT, font='Open Sans', pos=(0, 0), height=0.05)
    if duration:
        localClock = core.Clock()
        localClock.reset()
        while localClock.getTime() < duration:
            text_stim.draw()
            win.flip()
            CHECK_for_QUIT()
    if wait_key:
        ContinueRoutine = True
        resp = keyboard.Keyboard()
        while ContinueRoutine:
            text_stim.setAutoDraw(True)
            if resp.getKeys(keyList=[wait_key], waitRelease=False):
                ContinueRoutine = False
                text_stim.setAutoDraw(False)
                core.wait(0.2)
            win.flip()
            CHECK_for_QUIT()
    if countdown:
        while countdown > 0:
            text=f'{TEXT}\n\n{countdown}秒後に動画が始まります。'
            text_stim = visual.TextStim(
                win, text=text, font='Open Sans', pos=(0, 0), height=0.05)
            text_stim.draw()
            win.flip()
            time.sleep(1) # using time library
            countdown -= 1
            CHECK_for_QUIT()
            
def routine_PPORT():
    pport = parallel.ParallelPort(address='0x3FD8')

    # Create some handy timers
    globalClock = core.Clock()  # to track the time since experiment started
    routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

    # --- Prepare to start Routine "trial" ---
    continueRoutine = True
    # update component parameters for each repeat
    # keep track of which components have finished
    trialComponents = [pport]
    for thisComponent in trialComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1

    # --- Run Routine "trial" ---
    routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 0.05:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # *pport* updates
        
        # if pport is starting this frame...
        if pport.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            pport.frameNStart = frameN  # exact frame index
            pport.tStart = t  # local t and not account for scr refresh
            pport.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(pport, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.addData('pport.started', t)
            # update status
            pport.status = STARTED
            pport.status = STARTED
            win.callOnFlip(pport.setData, int(1))
        
        # if pport is stopping this frame...
        if pport.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > pport.tStartRefresh + .05-frameTolerance:
                # keep track of stop time/frame for later
                pport.tStop = t  # not accounting for scr refresh
                pport.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.addData('pport.stopped', t)
                # update status
                pport.status = FINISHED
                win.callOnFlip(pport.setData, int(0))
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
            if eyetracker:
                eyetracker.setConnectionState(False)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()

    # --- Ending Routine "trial" ---
    for thisComponent in trialComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    if pport.status == STARTED:
        win.callOnFlip(pport.setData, int(0))
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-0.050000)


def routine_MOVIE(pause_duration=3):
    if expInfo['Condition (1/2)'] == '1':
        movies = KazuMovies
    if expInfo['Condition (1/2)'] == '2':
        movies = ImiMovies
    x = 1.35
    y = x * 3/4
    movie = visual.MovieStim(
        win, filename=movies, loop=False, noAudio=True, 
        pos=(0, 0), size=(x, y), units=win.units) # size 4:3
    # --- Prepare to start Routine "Movies" ---
    movie.status = NOT_STARTED
    continueRoutine = True
    localClock = core.Clock()
    localClock.reset()
    while continueRoutine:
        if movie.status == NOT_STARTED:
            movie.setAutoDraw(True)
            while localClock.getTime() < pause_duration:
                win.flip()
                movie.pause()
                CHECK_for_QUIT()
            routine_PPORT() # send a signal when a movie starts 
            movie.play()
        if movie.isFinished:  # force-end the routine
            routine_PPORT() # send a signal when a movie finished    
            continueRoutine = False
            movie.stop()
            core.wait(0.4)
            movie.setAutoDraw(False)
        win.flip()
        CHECK_for_QUIT()

def routine_ANSWER(feedback=False):        
    text_ans = visual.TextStim(
        win, text='', font='Open Sans', pos=(0, 0), height=0.05,
        color='white', opacity=None)
    resp = keyboard.Keyboard()
    resp.keys = []
    
    if expInfo['Condition (1/2)'] == '1':
        text_ans.setText('モデルの動作の回数を、1~9のキーで答えてください。') 
        keylist = ['1','2','3', '4', '5', '6', '7', '8', '9']
    if expInfo['Condition (1/2)'] == '2':
        text_ans.setText(ImiAnswers)
        keylist = ['1','2','3']

    continueRoutine = True
    while continueRoutine:
        text_ans.setAutoDraw(True)
        resp.keys.extend(resp.getKeys(keyList=keylist, waitRelease=False))
        if resp.keys:
            text_ans.setAutoDraw(False)
            got_keys = resp.keys[0].name
            continueRoutine = False    
        win.flip()
        CHECK_for_QUIT()

    if feedback == True:
        # give a feedback to a participant
        if expInfo['Condition (1/2)'] == '1':
            text = f'あなたが選んだのは{got_keys}です。\n\n{KazuFeedback}'
        if expInfo['Condition (1/2)'] == '2':
            text = f'あなたが選んだのは{got_keys}です。\n\n{ImiFeedback}'
        feedback = visual.TextStim(
            win=win, name='text', text=text,
            font='Open Sans', pos=(0, 0), height=0.05)
        localClock = core.Clock()
        localClock.reset()
        while localClock.getTime() < 3:
            feedback.draw()
            win.flip()
            CHECK_for_QUIT()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# "First_Text" ---
if expInfo['Condition (1/2)'] == '1':
    routine_TEXT('それではこれからカウント条件を始めます。\n\n準備ができたらスペースキーを押してください。', 
    wait_key='space')
if expInfo['Condition (1/2)'] == '2':
    routine_TEXT('それではこれから意味条件を始めます。\n\n準備ができたらスペースキーを押してください。', 
    wait_key='space')

#set up handler to look after randomisation of conditions etc
loop = data.TrialHandler(
    nReps=1, method='random', extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions("loopTemplate1.xlsx"),
    seed=None)
thisExp.addLoop(loop)  # add the loop to the experiment
thisLoop = loop.trialList[0] # so we can initialise stimuli with some values

num_trial = 1
# Starting a loop
for thisLoop in loop:
    for paramName in thisLoop:
        exec('{} = thisLoop[paramName]'.format(paramName))
        
    routine_TEXT(f'これは{num_trial}回目の試行です。', duration=2)
    
    if expInfo['Condition (1/2)'] == '1':
        routine_TEXT('モデルの動作が何回かを数えてください。', countdown=5)
        routine_MOVIE()
        routine_ANSWER(feedback=False)
    if expInfo['Condition (1/2)'] == '2':
        routine_TEXT(f'{ImiCues}', countdown=7)
        routine_MOVIE()

    routine_TEXT('次に、動画を見ながらモデルのまねをしてください。', countdown=5) 
    routine_MOVIE()    
    
    if expInfo['Condition (1/2)'] == '1':
        routine_TEXT('それではもう一度、\nモデルの動作が何回かを数えてください。', countdown=5)
    if expInfo['Condition (1/2)'] == '2':
        routine_TEXT(f'それではもう一度、\n{ImiCues}', countdown=7)
    routine_MOVIE()   
    routine_ANSWER(feedback=True)
    num_trial += 1
# completd 'loop'

# "Fifth text"
if expInfo['Condition (1/2)'] == '1':
    routine_TEXT('これでこの実験条件は終了です。\n次の実験条件が始まるまで休憩をしてください。', duration=5)
if expInfo['Condition (1/2)'] == '2':
    routine_TEXT('これで今日の実験は終了です。\n本日はご協力いただき、本当にありがとうございました。', duration=3)

# --- End experiment ---
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
