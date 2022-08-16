/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2018 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "LoadBalancer.H"

//更新负载平衡的成员状态
void
Foam::LoadBalancer::updateState(
    const DynamicList<ChemistryProblem>& problems)
{
    auto myLoad = computeLoad(problems); //基类，计算负载，返回一个用于描述负载的结构体
    auto allLoads = allGather(myLoad); //基类，得到完整负载列表
    auto operations = getOperations(allLoads, myLoad); //计算操作，返回一个用于描述操作的结构体列表
    auto info = operationsToInfo(operations, problems, myLoad); //返回基类结构体

    setState(info); //设置基类成员变量
}

/*
 * 返回BalancerState结构体
 * 如果是sender，填充目的地，时间，计数器，问题数量
 * 如果是receiver，填充源地址，留存问题数量选择默认
 */
// 里面的time指代CPU时间
Foam::LoadBalancerBase::BalancerState
Foam::LoadBalancer::operationsToInfo(
    const std::vector<Operation>&        operations,
    const DynamicList<ChemistryProblem>& problems,
    const ChemistryLoad&                 myLoad)
{
    BalancerState info; //基类结构体，标记收发基本信息

    if(isSender(operations, myLoad.rank)) //要求每一个operation都是sender
    {
        double sum = 0.0;
        std::vector<double> times;
        for(const auto& op : operations) //将operation中的数据迁移到balancerstate中
        {
            info.destinations.push_back(op.to);
            sum += op.value;
            times.push_back(op.value);
        }
        info.nProblems = timesToProblemCounts(times, problems); //将时间转换为问题数量

        //todo 这句话如何理解？
        label total = std::accumulate(info.nProblems.begin(), info.nProblems.end(), 0); // counts of problems sent elsewhere
        info.nRemaining = problems.size() - total; //问题总量减去已经发送的问题数量=剩余在本地的问题数量
    }

    // receiver
    else //其他sender发给自己，填写源
    {
        for(const auto& op : operations)
        {
            info.sources.push_back(op.from);
        }
        info.nProblems = {};
        info.nRemaining = problems.size();
    }


    return info;
}

//把CPU时间转换为问题数量
std::vector<Foam::label>
Foam::LoadBalancer::timesToProblemCounts(
    const std::vector<scalar>&           times,
    const DynamicList<ChemistryProblem>& problems)
{

    std::vector<int> counts;
    counts.reserve(times.size() + 1);
    auto begin = problems.begin(); // 第一个问题的位置

    for(const auto& time : times)
    {
        scalar sum(0);
        auto operation = [&](const ChemistryProblem& problem) //求和
        {
            sum += problem.cpuTime;
            return sum <= time;
        };
        auto count = count_while(begin, problems.end(), operation);
        begin += count;
        counts.push_back(count);
    }

    return counts;
}

// 负载均衡主逻辑
std::vector<Foam::LoadBalancer::Operation>
Foam::LoadBalancer::getOperations(
    DynamicList<ChemistryLoad>& loads, const ChemistryLoad& myLoad)
{

    double globalMean = getMean(loads); //计算列表的平均负载

    std::vector<Operation> operations;

    std::sort(loads.begin(), loads.end()); //按负载排序

    auto sender = loads.end() - 1; //最后一个负载大于平均负载的CPU
    auto receiver = loads.begin(); //第一个负载小于平均负载的CPU

    while(sender != receiver) //所有人负载都平均，当过receiver就不会变成sender了，反之也是
    {
        //先让其中一个核到平均值
        double send_value = std::min(
            sender->value - globalMean,
            globalMean - receiver->value); //计算高负载和平均值的差值，低负载和平均值的差值，取最小值作为发送量

        Operation operation{sender->rank, receiver->rank, send_value}; //构造发送和接收操作
        if(sender->rank == myLoad.rank || receiver->rank == myLoad.rank)
        {
            operations.push_back(operation); //如果是自己的CPU，则添加操作
        }
        sender->value -= send_value; //更新负载
        receiver->value += send_value; //更新负载

        if(std::abs(sender->value - globalMean) < SMALL)
        {
            sender--; //如果负载小于平均负载，则移动到下一个负载
        }
        else
        {
            receiver++; //如果负载大于平均负载，则移动到下一个负载
        }
    }

    // explicitly filter very small operations
    std::vector<Operation> large;
    for(const auto& op : operations)
    {
        if(op.value > 0.01 * globalMean) //如果发送量大于1%平均负载
        {
            large.push_back(op); //那么就把操作添加到大操作列表中
        }
    }

    runtime_assert(
        !((isSender(operations, myLoad.rank) &&
           isReceiver(operations, myLoad.rank))),
        "Only sender or receiver should be possible.");

    runtime_assert(
        std::abs(getMean(loads) - globalMean) < 1E-7, "Vanishing load");

    return large; //返回大操作列表
}

//perform load balance with Redez-vous algorithm
std::vector<Foam::LoadBalancer::Operation>
Foam::LoadBalancer::getOperationsRedezVous(int &loads, const ChemistryLoad &myLoad)
{
    double globalMean = getMean(loads); //calculate the mean load
    std::vector<Operation> operations;
    std::sort(loads.begin(), loads.end()); //sort the loads
    auto sender = loads.end() - 1;
    auto receiver = loads.begin();

    while(sender != receiver)
    {
        double send_value = (sender->value - receiver->value) / 2; // calculate the send value
        Operation operation{sender->rank, receiver->rank, send_value};
        if(sender->rank == myLoad.rank || receiver->rank == myLoad.rank)
        {
            operations.push_back(operation); // if send or recv rank related to my rank, add to operations
        }
        sender->value -= send_value; // update the load
        receiver->value += send_value; // update the load
        sender--;
        receiver++;
    }

    // explicitly filter very small operations
    std::vector<Operation> large;
    for(const auto& op:operations)
    {
        if(op.value > 0.01 * globalMean) //if send value is larger than 1% of global mean
        {
            large.push_back(op); // add to large operations
        }
    }
    runtime_assert(
            !((isSender(operations, myLoad.rank) &&
            isReceiver(operations, myLoad.rank))),
            "Only sender or receiver should be possible.");

    runtime_assert(
            std::abs(getMean(loads) - globalMean) < 1E-7, "Vanishing load");

    return large; //return large operations
}


// 查看操作中的源是否是自己，如果是则为接收者，否则为发送者
bool
Foam::LoadBalancer::isSender(
    const std::vector<Operation>& operations, int rank)
{
    if(operations.size() == 0)
    {
        return false;
    }

    for(const auto& op : operations)
    {
        if(op.from != rank)
        {
            return false;
        }
    }
    return true;
}

// 查看操作中的源是否是自己，如果是则为接收者，否则为发送者
bool
Foam::LoadBalancer::isReceiver(
    const std::vector<Operation>& operations, int rank)
{
    for(const auto& op : operations)
    {
        if(op.to != rank)
        {
            return false;
        }
    }
    return true;
}

